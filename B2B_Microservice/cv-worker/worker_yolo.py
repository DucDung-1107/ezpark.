#!/usr/bin/env python3
"""
worker_yolo.py — Full rewritten

Chức năng:
- Phát hiện xe bằng YOLOv8 (ultralytics)
- Theo dõi đơn giản (centroid tracker)
- Phát hiện crossing qua một đường ngang -> auto enter/exit
- ANPR: Cloud (PlateRecognizer) nếu có API key, fallback local via easyocr (nếu cài)
- Endpoint HTTP POST /event để trigger thủ công
- Config via environment variables or .env

Cách chạy (local, không docker):
1) tạo virtualenv, activate
2) pip install -r requirements (xem dưới)
3) export các biến môi trường (hoặc tạo .env)
   - VIDEO_PATH=/full/path/sample.mp4
   - CV_DEBUG=1
   - PLATE_RECOGNIZER_API_KEY=... (nếu có)
   - ENTRY_LINE=0.5
4) python cv-worker/worker_yolo.py --video "$VIDEO_PATH"

Lưu ý:
- Nếu không có PLATE_RECOGNIZER_API_KEY, worker sẽ fallback sang easyocr nếu cài.
- Thông số chính available via env: DETECT_CONF, DETECT_IOU, FRAME_SKIP, MATCH_DISTANCE, CLEANUP_SECONDS, COUNT_COOLDOWN, ANPR_CONF_THRESHOLD.
"""

import re
import os
import sys
import time
import io
import threading
import argparse
from collections import deque

# optional dotenv (load .env if present)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import cv2
import numpy as np
import requests
from PIL import Image
from flask import Flask, request, jsonify

# optional ultralytics YOLO
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# optional easyocr
try:
    import easyocr
except Exception:
    easyocr = None

# --------------------- Configuration (env / defaults) ---------------------
VIDEO_PATH = os.environ.get("VIDEO_PATH", "/video/input.mp4")
BACKEND_URL = os.environ.get("BACKEND_URL", "")   # nếu dùng backend thì set
SITE_ID = os.environ.get("SITE_ID", "site_demo_1")

PLATE_API_KEY = os.environ.get("PLATE_RECOGNIZER_API_KEY", "").strip()
PLATE_API_URL = os.environ.get("PLATE_RECOGNIZER_URL", "https://api.platerecognizer.com/v1/plate-reader/")

FRAME_SKIP = int(os.environ.get("FRAME_SKIP", "1"))
CV_DEBUG = os.environ.get("CV_DEBUG", "0") == "1"

DETECT_CONF = float(os.environ.get("DETECT_CONF", "0.35"))
DETECT_IOU = float(os.environ.get("DETECT_IOU", "0.45"))

MAX_TRACKED_OBJECTS = int(os.environ.get("MAX_TRACKED_OBJECTS", "120"))
MATCH_DISTANCE = int(os.environ.get("MATCH_DISTANCE", "60"))   # px for centroid match
CLEANUP_SECONDS = float(os.environ.get("CLEANUP_SECONDS", "2.5"))

ANPR_CONF_THRESHOLD = float(os.environ.get("ANPR_CONF_THRESHOLD", "0.55"))

# crossing config
ENTRY_LINE = os.environ.get("ENTRY_LINE", "0.5")  # fraction of height (0..1) or absolute px
ENTRY_DIRECTION_IN = os.environ.get("ENTRY_DIRECTION_IN", "down")  # "down" or "up"
COUNT_COOLDOWN = float(os.environ.get("COUNT_COOLDOWN", "3.0"))  # seconds to avoid duplicate counts per object

# vehicle classes to accept (ultralytics names)
VEHICLE_CLASS_NAMES = set(
    name.lower() for name in os.environ.get("VEHICLE_CLASSES", "car,motorcycle,bus,truck,bicycle").split(",")
)

# --------------------- Globals ---------------------
tracker_objects = {}      # oid -> (cx,cy)
tracker_bboxes = {}       # oid -> (x,y,w,h)
tracker_history = {}      # oid -> deque of (cx,cy)
tracker_last_seen = {}    # oid -> timestamp
tracker_counted_at = {}   # oid -> last counted timestamp (to avoid double)
_next_object_id_lock = threading.Lock()
next_object_id = 0
ct_lock = threading.Lock()

allocated_spots = deque()
current_allocated = set()

total_entered = 0
total_exited = 0
counters_lock = threading.Lock()

app = Flask(__name__)

_easyocr_reader = None


# regex đơn giản ưu tiên cho biển VN (heuristic): 2-3 digits, dash, 1-2 letters, space, groups of digits (dot optional)
_VN_PLATE_REGEX = re.compile(r'([0-9]{2,3})\s*[-]?\s*([A-Z0-9]{1,3})\s*[-]?\s*([0-9]{3}[.\s]?[0-9]{2,3})', re.I)

def _normalize_plate_text(s: str) -> str:
    if not s: 
        return s
    # keep digits and letters and dash
    s2 = re.sub(r'[^A-Za-z0-9\-\.]', '', s).upper()
    # replace multiple dots/spaces
    s2 = re.sub(r'\.+', '.', s2)
    return s2

def _plate_score_by_regex(text: str):
    """Trả điểm cao nếu khớp regex VN plate"""
    if not text:
        return 0.0
    txt = _normalize_plate_text(text)
    if _VN_PLATE_REGEX.search(txt):
        return 1.0
    # else small bonus for many digits/letters
    alnum = len(re.sub(r'[^A-Za-z0-9]', '', txt))
    return min(0.6, 0.1 * alnum)

def _generate_preprocessing_variants(img_bgr):
    """Trả về list các biến thể ảnh (BGR) để thử OCR"""
    imgs = []
    # original (resized)
    H, W = img_bgr.shape[:2]
    scale = 1.0
    if max(H, W) < 300:
        scale = 1.6
    img0 = cv2.resize(img_bgr, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_CUBIC)
    imgs.append(img0)

    # convert to HSV and CLAHE on V/L channel
    try:
        lab = cv2.cvtColor(img0, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        imgs.append(cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR))
    except Exception:
        pass

    # gamma variants
    for gamma in (0.6, 0.8, 1.2, 1.6):
        invGamma = 1.0 / gamma
        table = np.array([((i/255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
        imgs.append(cv2.LUT(img0, table))

    # sharpened
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    try:
        imgs.append(cv2.filter2D(img0, -1, kernel))
    except Exception:
        pass

    # grayscale + adaptive threshold variants (convert to BGR for easyocr which accepts color)
    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    for block in (31, 41):
        try:
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block, 9)
            imgs.append(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))
        except Exception:
            pass

    # median blur removal of small noise
    imgs.append(cv2.medianBlur(img0, 3))
    return imgs

def _detect_plate_rectangle(img_bgr):
    """
    Tìm contour hình chữ nhật có tỉ lệ tương tự biển số (chiều ngang > chiều dọc, ratio ~ 1.5..4)
    Trả bbox (x,y,w,h) nếu tìm được, else None.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # equalize + reduce noise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # edge detect
    edges = cv2.Canny(gray, 50, 150)
    # dilate để nối các cạnh
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0.0
    for c in cnts:
        x,y,ww,hh = cv2.boundingRect(c)
        if ww < 0.2*w or hh < 0.03*h:   # quá nhỏ so với crop
            continue
        ar = ww / float(hh + 1e-6)
        # biển số thường rộng hơn cao, ar khoảng 1.2 - 5 phụ thuộc mẫu
        if ar < 1.2 or ar > 6.0:
            continue
        area = cv2.contourArea(c)
        solidity = area / float(ww*hh + 1e-6)
        if solidity < 0.3:
            continue
        # ưu tiên gần tâm dưới crop (thường biển ở đáy xe)
        cy = y + hh/2
        center_y = h/2
        center_score = 1.0 - abs(cy - center_y) / (h/2)
        score = center_score * solidity * (ar/6.0)
        if score > best_score:
            best_score = score
            best = (x,y,ww,hh)
    return best

def recognize_plate_local_easyocr_improved(image_bgr):
    """
    Cải tiến local OCR: 
    1) thử detect_plate_rectangle để crop nhỏ vùng plate
    2) nếu tìm được, tạo variants preprocess cho region
    3) chạy easyocr trên mỗi biến thể, chọn kết quả có score cao nhất (confidence + regex match)
    4) nếu không tìm region, thử trên toàn crop với các biến thể
    Trả (best_plate_text_or_None, best_conf_score [0..1])
    """
    if easyocr is None:
        return None, 0.0

    best_plate = None
    best_score = 0.0

    # 1) try detect plate rectangle
    plate_rect = _detect_plate_rectangle(image_bgr)
    candidate_regions = []
    if plate_rect:
        x,y,ww,hh = plate_rect
        # pad nhẹ
        px = int(ww * 0.12); py = int(hh * 0.15)
        x0 = max(0, x - px); y0 = max(0, y - py)
        x1 = min(image_bgr.shape[1], x + ww + px); y1 = min(image_bgr.shape[0], y + hh + py)
        region = image_bgr[y0:y1, x0:x1]
        candidate_regions.append(region)
    else:
        # fallback: assume plate is lower half of crop, take multiple ROIs
        H = image_bgr.shape[0]
        candidate_regions.append(image_bgr[int(H*0.4):H, :])    # bottom 60%
        candidate_regions.append(image_bgr[int(H*0.5):H, :])    # bottom 50%

    # init easyocr reader lazily
    global _easyocr_reader
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(['en'], gpu=False)

    # try each candidate region with variants
    for region in candidate_regions:
        variants = _generate_preprocessing_variants(region)
        for var in variants:
            try:
                # easyocr returns list of (bbox, text, conf)
                results = _easyocr_reader.readtext(var)
            except Exception as e:
                # ignore variant if error
                continue
            for _, txt, conf in results:
                txt_clean = _normalize_plate_text(txt)
                # compute score combining ocr confidence and regex match
                rscore = float(conf if conf is not None else 0.0)
                regex_bonus = _plate_score_by_regex(txt_clean)
                score = rscore * 0.6 + regex_bonus * 0.4
                # small length penalty (prefer medium length)
                if len(re.sub(r'[^A-Za-z0-9]', '', txt_clean)) < 4:
                    score *= 0.6
                if score > best_score:
                    best_score = score
                    best_plate = txt_clean

    # clamp confidence to 0..1
    best_score = max(0.0, min(1.0, best_score))
    return (best_plate, best_score)

# --------------------- Helpers ---------------------
def safe_print(*args, **kwargs):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), *args, **kwargs)

def next_object_id_inc():
    global next_object_id
    with _next_object_id_lock:
        oid = next_object_id
        next_object_id += 1
    return oid

# --------------------- ANPR (cloud + local) ---------------------
def recognize_plate_cloud(image_bgr):
    """Call PlateRecognizer (cloud). Return (plate, conf) or (None, 0.0)."""
    if not PLATE_API_KEY:
        return None, 0.0
    try:
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=80)
        jbytes = buf.getvalue()
        headers = {"Authorization": f"Token {PLATE_API_KEY}"}
        files = {"upload": ("frame.jpg", jbytes, "image/jpeg")}
        r = requests.post(PLATE_API_URL, headers=headers, files=files, timeout=10)
        if r.status_code == 200:
            data = r.json()
            results = data.get("results") or data.get("data") or []
            if results:
                first = results[0]
                plate = first.get("plate") or first.get("plate_number") or first.get("vehicle_plate") or ""
                conf = first.get("score") or first.get("confidence") or 0.0
                try:
                    return plate.strip(), float(conf)
                except Exception:
                    return plate.strip(), 0.0
    except Exception as e:
        safe_print("[ANPR] cloud error:", e)
    return None, 0.0

def recognize_plate_local_easyocr(image_bgr):
    """Local OCR via easyocr. Return (plate, conf)."""
    global _easyocr_reader
    if easyocr is None:
        return None, 0.0
    try:
        if _easyocr_reader is None:
            # language 'en' is fine for alphanumeric plates; set gpu=True if available
            _easyocr_reader = easyocr.Reader(['en'], gpu=False)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        h, w = gray.shape[:2]
        if max(h, w) < 200:
            gray = cv2.resize(gray, (0,0), fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 51, 9)
        results = _easyocr_reader.readtext(th)
        best = None
        for bbox, text, conf in results:
            txt = "".join(ch for ch in text if ch.isalnum())
            if len(txt) >= 4:
                if best is None or conf > best[1]:
                    best = (txt, conf)
        if best:
            return best[0], float(best[1])
    except Exception as e:
        safe_print("[ANPR] easyocr error:", e)
    return None, 0.0

def recognize_plate(image_bgr):
    if PLATE_API_KEY:
        p, c = recognize_plate_cloud(image_bgr)
        if p and c >= ANPR_CONF_THRESHOLD:
            return p, c
    return recognize_plate_local_easyocr_improved(image_bgr)

# --------------------- Backend helpers (slot update) ---------------------
def post_slot_update(site_id, spot_id, status, image_url=None, plate=None):
    payload = {
        "event_id": f"evt_{spot_id}_{int(time.time())}",
        "site_id": site_id,
        "spot_id": spot_id,
        "status": status,
        "image_url": image_url,
        "plate": plate,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if not BACKEND_URL:
        safe_print("[worker] no BACKEND_URL set, would send:", payload)
        return None
    try:
        r = requests.post(f"{BACKEND_URL}/api/events/slot_update", json=payload, timeout=6)
        safe_print("[worker] POST slot_update", spot_id, status, "plate:", plate, "->", r.status_code)
        return r
    except Exception as e:
        safe_print("[worker] Error POST slot_update:", e)
        return None

def find_free_spot_via_backend(site_id):
    if not BACKEND_URL:
        return None
    try:
        r = requests.get(f"{BACKEND_URL}/api/spots?site_id={site_id}", timeout=4)
        if r.status_code == 200:
            for s in r.json().get("spots", []):
                if s.get("status") in (None, "", "free"):
                    return s["id"]
    except Exception as e:
        safe_print("[worker] error fetching spots:", e)
    return None

def find_session_by_plate(plate):
    if not (BACKEND_URL and plate):
        return None
    try:
        r = requests.get(f"{BACKEND_URL}/api/sessions?plate={plate}", timeout=5)
        if r.status_code == 200:
            sessions = r.json().get("sessions", [])
            for s in sessions:
                if s.get("status") == "active":
                    return s
    except Exception as e:
        safe_print("[worker] error fetching sessions by plate:", e)
    return None

# --------------------- Event processing (enter/exit) ---------------------
def process_event_by_bbox(direction, bbox, frame):
    """
    direction: 'enter'|'exit'
    bbox: (x,y,w,h)
    frame: BGR numpy array
    """
    x, y, w, h = bbox
    pad_x = int(w * 0.6)
    pad_y = int(h * 0.8)
    x0 = max(0, x - pad_x); y0 = max(0, y - pad_y)
    x1 = min(frame.shape[1], x + w + pad_x); y1 = min(frame.shape[0], y + h + pad_y)
    crop = frame[y0:y1, x0:x1]
    safe_print("[worker] process_event", direction, "crop_size=", crop.shape)

    plate, conf = recognize_plate(crop)
    safe_print("[worker] ANPR result:", plate, conf)

    used_plate = None
    if plate and conf >= ANPR_CONF_THRESHOLD:
        used_plate = plate

    # business logic: allocate/free spot (fallback simple FIFO)
    if direction == "enter":
        if used_plate:
            spot_id = find_free_spot_via_backend(SITE_ID)
            if spot_id:
                allocated_spots.append(spot_id)
                current_allocated.add(spot_id)
                post_slot_update(SITE_ID, spot_id, "occupied", image_url=None, plate=used_plate)
            else:
                safe_print("[worker] enter: no free spot (backend)")
        else:
            spot_id = find_free_spot_via_backend(SITE_ID)
            if spot_id:
                allocated_spots.append(spot_id)
                current_allocated.add(spot_id)
                post_slot_update(SITE_ID, spot_id, "occupied", image_url=None, plate=None)
            else:
                safe_print("[worker] enter: cannot allocate (no plate, no free spot)")
    else:  # exit
        if used_plate:
            sess = find_session_by_plate(used_plate)
            if sess:
                spot_id = sess.get("spot_id")
                if spot_id:
                    post_slot_update(SITE_ID, spot_id, "free", image_url=None, plate=used_plate)
                    try:
                        allocated_spots.remove(spot_id)
                        current_allocated.discard(spot_id)
                    except Exception:
                        pass
                else:
                    if allocated_spots:
                        sid = allocated_spots.popleft()
                        current_allocated.discard(sid)
                        post_slot_update(SITE_ID, sid, "free", image_url=None, plate=used_plate)
                    else:
                        safe_print("[worker] exit: session has no spot_id and no allocated fallback")
            else:
                if allocated_spots:
                    sid = allocated_spots.popleft()
                    current_allocated.discard(sid)
                    post_slot_update(SITE_ID, sid, "free", image_url=None, plate=used_plate)
                else:
                    # fallback: free first occupied from backend
                    if BACKEND_URL:
                        try:
                            r = requests.get(f"{BACKEND_URL}/api/spots?site_id={SITE_ID}", timeout=4)
                            if r.status_code == 200:
                                for s in r.json().get("spots", []):
                                    if s.get("status") == "occupied":
                                        post_slot_update(SITE_ID, s["id"], "free", image_url=None, plate=used_plate)
                                        break
                        except Exception as e:
                            safe_print("[worker] exit fallback error:", e)
        else:
            if allocated_spots:
                sid = allocated_spots.popleft()
                current_allocated.discard(sid)
                post_slot_update(SITE_ID, sid, "free", image_url=None, plate=None)
            else:
                safe_print("[worker] exit: no plate and no allocated spots")

# --------------------- HTTP endpoint (manual triggers) ---------------------
@app.route("/event", methods=["POST"])
def http_event():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid json"}), 400
    oid = data.get("object_id")
    direction = data.get("direction")
    if oid is None or direction not in ("enter", "exit"):
        return jsonify({"error": "invalid payload"}), 400
    with ct_lock:
        if oid not in tracker_bboxes:
            return jsonify({"error":"object_id not found"}), 404
        bbox = tracker_bboxes[oid]
        frame_snapshot = globals().get("latest_frame")
        if frame_snapshot is None:
            return jsonify({"error":"no frame available"}), 500
        threading.Thread(target=process_event_by_bbox, args=(direction, bbox, frame_snapshot.copy()), daemon=True).start()
    return jsonify({"status":"accepted","object_id":oid,"direction":direction}), 202

# --------------------- Detection + Tracking + Auto-crossing ---------------------
def run_video_loop(video_path):
    global total_entered, total_exited

    if YOLO is None:
        safe_print("[worker] ultralytics not installed. pip install ultralytics")
        sys.exit(1)

    safe_print("[worker] Loading YOLO model (yolov8n.pt) — may download weights if missing")
    try:
        model = YOLO("yolov8n.pt")
    except Exception as e:
        safe_print("[worker] Error loading YOLO:", e)
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        safe_print("[worker] cannot open video:", video_path)
        sys.exit(1)

    frame_idx = 0
    h = None; w = None
    # compute entry_line once we know frame size
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.2)
            continue

        frame_idx += 1
        if frame_idx % max(1, FRAME_SKIP) != 0:
            continue

        h, w = frame.shape[:2]
        # compute line_y pixel coordinate
        try:
            entry_val = float(ENTRY_LINE)
            if 0.0 < entry_val <= 1.0:
                line_y = int(h * entry_val)
            else:
                line_y = int(entry_val)
        except Exception:
            line_y = int(h * 0.5)

        globals()["latest_frame"] = frame

        # run YOLO detection
        try:
            results = model.predict(frame, conf=DETECT_CONF, iou=DETECT_IOU, device="cpu", verbose=False)
        except Exception as e:
            safe_print("[worker] YOLO inference error:", e)
            time.sleep(0.2)
            continue

        preds = results[0]
        boxes = getattr(preds, "boxes", None)
        rects = []
        names = getattr(model, "names", {}) or {}

        if boxes is not None and len(boxes) > 0:
            # robust extraction of arrays (cpu/gpu variations)
            try:
                xyxy = boxes.xyxy.cpu().numpy()
            except Exception:
                xyxy = np.array(boxes.xyxy)
            try:
                confs = boxes.conf.cpu().numpy()
            except Exception:
                confs = np.array(boxes.conf)
            try:
                cls = boxes.cls.cpu().numpy().astype(int)
            except Exception:
                cls = np.array(boxes.cls).astype(int)

            for bb, conf, c in zip(xyxy, confs, cls):
                class_name = names.get(int(c), str(c)).lower()
                if class_name in VEHICLE_CLASS_NAMES:
                    x1, y1, x2, y2 = map(int, bb[:4])
                    wbox = x2 - x1; hbox = y2 - y1
                    if wbox <= 8 or hbox <= 8:
                        continue
                    rects.append((x1, y1, wbox, hbox))

        # compute centroids
        new_centroids = []
        for (x, y, wbox, hbox) in rects:
            cx = int(x + wbox / 2)
            cy = int(y + hbox / 2)
            new_centroids.append((cx, cy, (x, y, wbox, hbox)))

        with ct_lock:
            matched_new = set()
            # match existing trackers: nearest centroid within MATCH_DISTANCE
            for idx, (cx, cy, bbox_xywh) in enumerate(new_centroids):
                for oid, cent in list(tracker_objects.items()):
                    dist = np.hypot(cent[0] - cx, cent[1] - cy)
                    if dist < MATCH_DISTANCE:
                        tracker_objects[oid] = (cx, cy)
                        tracker_bboxes[oid] = bbox_xywh
                        tracker_history.setdefault(oid, deque(maxlen=30)).append((cx, cy))
                        tracker_last_seen[oid] = time.time()
                        matched_new.add(idx)
                        break

            # register unmatched
            for idx, (cx, cy, bbox_xywh) in enumerate(new_centroids):
                if idx in matched_new:
                    continue
                if len(tracker_objects) >= MAX_TRACKED_OBJECTS:
                    continue
                oid = next_object_id_inc()
                tracker_objects[oid] = (cx, cy)
                tracker_bboxes[oid] = bbox_xywh
                tracker_history[oid] = deque(maxlen=30)
                tracker_history[oid].append((cx, cy))
                tracker_last_seen[oid] = time.time()
                tracker_counted_at[oid] = 0.0
                safe_print(f"[worker] registered object {oid} at {(cx,cy)}")

            # detect crossing using last two centroids for each tracked object
            now = time.time()
            for oid, hist in list(tracker_history.items()):
                if len(hist) < 2:
                    continue
                prev = hist[-2]; cur = hist[-1]
                prev_y = prev[1]; cur_y = cur[1]
                # crossing condition (prev->cur crosses line_y)
                crossed = (prev_y < line_y <= cur_y) or (prev_y > line_y >= cur_y)
                if not crossed:
                    continue
                # avoid double count
                last_count = tracker_counted_at.get(oid, 0.0)
                if now - last_count < COUNT_COOLDOWN:
                    continue
                # determine logical direction
                if prev_y < cur_y:
                    direction = "enter" if ENTRY_DIRECTION_IN == "down" else "exit"
                else:
                    direction = "exit" if ENTRY_DIRECTION_IN == "down" else "enter"
                # update counters
                with counters_lock:
                    if direction == "enter":
                        total_entered += 1
                    else:
                        total_exited += 1
                    safe_print(f"[worker] auto-cross: oid={oid} direction={direction} total_entered={total_entered} total_exited={total_exited}")
                tracker_counted_at[oid] = now
                # trigger event processing (ANPR + backend) in background
                bbox_use = tracker_bboxes.get(oid)
                frame_snapshot = globals().get("latest_frame")
                if bbox_use is not None and frame_snapshot is not None:
                    threading.Thread(target=process_event_by_bbox, args=(direction, bbox_use, frame_snapshot.copy()), daemon=True).start()

            # cleanup disappeared objects
            to_remove = [oid for oid, ts in tracker_last_seen.items() if now - ts > CLEANUP_SECONDS]
            for oid in to_remove:
                tracker_objects.pop(oid, None)
                tracker_bboxes.pop(oid, None)
                tracker_history.pop(oid, None)
                tracker_last_seen.pop(oid, None)
                tracker_counted_at.pop(oid, None)

            # enforce max tracked objects
            if len(tracker_objects) > MAX_TRACKED_OBJECTS:
                items = sorted(tracker_last_seen.items(), key=lambda kv: kv[1])
                remove_count = len(tracker_objects) - MAX_TRACKED_OBJECTS
                for oid, _ in items[:remove_count]:
                    tracker_objects.pop(oid, None)
                    tracker_bboxes.pop(oid, None)
                    tracker_history.pop(oid, None)
                    tracker_last_seen.pop(oid, None)
                    tracker_counted_at.pop(oid, None)

        # debug overlays
        if CV_DEBUG:
            disp = frame.copy()
            # draw entry line
            cv2.line(disp, (0, line_y), (w, line_y), (0, 0, 255), 2)
            with ct_lock:
                for oid, (cx, cy) in tracker_objects.items():
                    bbox = tracker_bboxes.get(oid)
                    if bbox:
                        x, y, wb, hb = bbox
                        cv2.rectangle(disp, (x, y), (x + wb, y + hb), (0, 255, 0), 2)
                        cv2.putText(disp, f"ID:{oid}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(disp, f"Entered:{total_entered}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(disp, f"Exited:{total_exited}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("worker_yolo", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if CV_DEBUG:
        cv2.destroyAllWindows()

# --------------------- CLI simulate ---------------------
def cli_simulate():
    safe_print("CLI simulate mode: 'l' list, 'e <id>' enter, 'x <id>' exit, 'q' quit")
    while True:
        cmd = input("> ").strip().split()
        if not cmd:
            continue
        if cmd[0] == "l":
            with ct_lock:
                safe_print("objects:", list(tracker_objects.keys()))
                for oid in tracker_objects.keys():
                    safe_print(oid, "bbox=", tracker_bboxes.get(oid), "centroid=", tracker_objects.get(oid))
        elif cmd[0] == "e" and len(cmd) > 1:
            try:
                oid = int(cmd[1])
                with ct_lock:
                    if oid in tracker_bboxes and globals().get("latest_frame") is not None:
                        process_event_by_bbox("enter", tracker_bboxes[oid], globals()["latest_frame"].copy())
                    else:
                        safe_print("object not found or no frame")
            except Exception as e:
                safe_print("err:", e)
        elif cmd[0] == "x" and len(cmd) > 1:
            try:
                oid = int(cmd[1])
                with ct_lock:
                    if oid in tracker_bboxes and globals().get("latest_frame") is not None:
                        process_event_by_bbox("exit", tracker_bboxes[oid], globals()["latest_frame"].copy())
                    else:
                        safe_print("object not found or no frame")
            except Exception as e:
                safe_print("err:", e)
        elif cmd[0] == "q":
            safe_print("exit CLI")
            break
        else:
            safe_print("unknown command")

# --------------------- Flask server starter ---------------------
def start_flask_thread():
    def run():
        app.run(host="0.0.0.0", port=5001, debug=False)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t

# --------------------- Main ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=VIDEO_PATH, help="path to video file")
    parser.add_argument("--cli", action="store_true", help="enable CLI simulate mode")
    args = parser.parse_args()

    # start web server for manual /event
    start_flask_thread()

    if args.cli:
        video_thread = threading.Thread(target=run_video_loop, args=(args.video,), daemon=True)
        video_thread.start()
        cli_simulate()
    else:
        try:
            run_video_loop(args.video)
        except KeyboardInterrupt:
            safe_print("stopped by user")

if __name__ == "__main__":
    main()