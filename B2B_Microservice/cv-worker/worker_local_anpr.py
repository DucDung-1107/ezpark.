#!/usr/bin/env python3
"""
export VIDEO_PATH=/Users/hoangquannguyen/Downloads/oto.mov
export CV_DEBUG=1
export ENTRY_LINE=0.5
export ANPR_CONF_THRESHOLD=0.35
python cv-worker/worker_local_anpr.py --video "$VIDEO_PATH"
"""

import os, sys, time, io, threading, argparse, re, csv
from collections import deque
from pathlib import Path

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

# ultralytics YOLO
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# easyocr local OCR
try:
    import easyocr
except Exception:
    easyocr = None

# ---------------- Config ----------------
VIDEO_PATH = os.environ.get("VIDEO_PATH", "/video/input.mp4")
FRAME_SKIP = int(os.environ.get("FRAME_SKIP", "1"))
CV_DEBUG = os.environ.get("CV_DEBUG", "0") == "1"
DETECT_CONF = float(os.environ.get("DETECT_CONF", "0.35"))
DETECT_IOU  = float(os.environ.get("DETECT_IOU", "0.45"))
MAX_TRACKED_OBJECTS = int(os.environ.get("MAX_TRACKED_OBJECTS", "120"))
MATCH_DISTANCE = int(os.environ.get("MATCH_DISTANCE", "60"))
CLEANUP_SECONDS = float(os.environ.get("CLEANUP_SECONDS", "2.5"))
COUNT_COOLDOWN = float(os.environ.get("COUNT_COOLDOWN", "3.0"))
ANPR_CONF_THRESHOLD = float(os.environ.get("ANPR_CONF_THRESHOLD", "0.35"))

ENTRY_LINE = os.environ.get("ENTRY_LINE", "0.5")
ENTRY_DIRECTION_IN = os.environ.get("ENTRY_DIRECTION_IN", "down")

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./evidence"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_CSV = OUTPUT_DIR / "events.csv"

VEHICLE_CLASSES = set(name.lower() for name in os.environ.get("VEHICLE_CLASSES", "car,motorcycle,bus,truck,bicycle").split(","))

# ---------------- Globals ----------------
tracker_objects = {}      # oid -> (cx,cy)
tracker_bboxes = {}       # oid -> (x,y,w,h)
tracker_history = {}      # oid -> deque((cx,cy))
tracker_last_seen = {}    # oid -> last seen ts
tracker_counted_at = {}   # oid -> last counted ts
_next_object_id_lock = threading.Lock()
next_object_id = 0
ct_lock = threading.Lock()

total_entered = 0
total_exited = 0
counters_lock = threading.Lock()

_easyocr_reader = None

app = Flask(__name__)

# regex heuristic for VN plates (simple)
_VN_PLATE_REGEX = re.compile(r'([0-9]{2,3})\s*[-]?\s*([A-Z0-9]{1,3})\s*[-]?\s*([0-9]{3}[.\s]?[0-9]{2,3})', re.I)

# ---------------- Helpers ----------------
def tprint(*args, **kwargs):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), *args, **kwargs)

def next_object_id_inc():
    global next_object_id
    with _next_object_id_lock:
        oid = next_object_id
        next_object_id += 1
    return oid

def _normalize_plate_text(s):
    if not s:
        return s
    s2 = re.sub(r'[^A-Za-z0-9\.\-]', '', s).upper()
    s2 = re.sub(r'\.+', '.', s2)
    return s2

def _plate_regex_score(text):
    if not text:
        return 0.0
    txt = _normalize_plate_text(text)
    if _VN_PLATE_REGEX.search(txt):
        return 1.0
    alnum = len(re.sub(r'[^A-Za-z0-9]', '', txt))
    return min(0.6, 0.1 * alnum)

# ---------------- Preprocessing variants & plate ROI detection ----------------
def _generate_variants(img_bgr):
    """Return list of BGR variants for OCR attempts."""
    imgs = []
    H, W = img_bgr.shape[:2]
    scale = 1.0
    if max(H, W) < 300:
        scale = 1.6
    img0 = cv2.resize(img_bgr, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_CUBIC)
    imgs.append(img0)
    # CLAHE on L channel
    try:
        lab = cv2.cvtColor(img0, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2,a,b))
        imgs.append(cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR))
    except Exception:
        pass
    # gamma variants (reduce/increase brightness)
    for gamma in (0.6, 0.8, 1.2, 1.6):
        invGamma = 1.0 / gamma
        table = np.array([((i/255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
        imgs.append(cv2.LUT(img0, table))
    # sharpen
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    try:
        imgs.append(cv2.filter2D(img0, -1, kernel))
    except Exception:
        pass
    # grayscale adaptive threshold variants
    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    for block in (31, 41):
        try:
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block, 9)
            imgs.append(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))
        except Exception:
            pass
    # median blur (de-noise)
    try:
        imgs.append(cv2.medianBlur(img0, 3))
    except Exception:
        pass
    return imgs

def _detect_plate_rect(img_bgr):
    """Find candidate rectangle in crop likely to be plate; return (x,y,w,h) or None"""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None; best_score = 0.0
    for c in cnts:
        x,y,ww,hh = cv2.boundingRect(c)
        if ww < 0.2*w or hh < 0.03*h:
            continue
        ar = ww / float(hh + 1e-6)
        if ar < 1.2 or ar > 6.0:
            continue
        area = cv2.contourArea(c)
        solidity = area / float(ww*hh + 1e-6)
        if solidity < 0.25:
            continue
        cy = y + hh/2
        center_y = h/2
        center_score = 1.0 - abs(cy - center_y) / (h/2)
        score = center_score * solidity * (ar/6.0)
        if score > best_score:
            best_score = score; best = (x,y,ww,hh)
    return best

# ---------------- local OCR improved ----------------
def recognize_plate_local_improved(image_bgr):
    """Try to detect plate ROI then run easyocr on many preprocessing variants. Return (plate_text or None, score 0..1)."""
    if easyocr is None:
        return None, 0.0
    global _easyocr_reader
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if you have GPU+torch configured

    best_plate = None
    best_score = 0.0

    H = image_bgr.shape[0]
    # prefer detected rect
    rect = _detect_plate_rect(image_bgr)
    candidate_regions = []
    if rect:
        x,y,ww,hh = rect
        px = int(ww * 0.12); py = int(hh * 0.15)
        x0 = max(0, x - px); y0 = max(0, y - py)
        x1 = min(image_bgr.shape[1], x + ww + px); y1 = min(image_bgr.shape[0], y + hh + py)
        candidate_regions.append(image_bgr[y0:y1, x0:x1])
    else:
        # fallback: bottom part (plates usually lower)
        candidate_regions.append(image_bgr[int(H*0.4):H, :])
        candidate_regions.append(image_bgr[int(H*0.5):H, :])

    for region in candidate_regions:
        variants = _generate_variants(region)
        for var in variants:
            try:
                results = _easyocr_reader.readtext(var)
            except Exception:
                continue
            for _, txt, conf in results:
                txt_clean = _normalize_plate_text(txt)
                rscore = float(conf if conf is not None else 0.0)
                regex_bonus = _plate_regex_score(txt_clean)
                score = rscore * 0.6 + regex_bonus * 0.4
                if len(re.sub(r'[^A-Za-z0-9]', '', txt_clean)) < 4:
                    score *= 0.6
                if score > best_score:
                    best_score = score; best_plate = txt_clean

    best_score = max(0.0, min(1.0, best_score))
    return best_plate, best_score

# ---------------- Event CSV logging ----------------
def append_event_csv(ts, oid, direction, plate, conf, crop_path):
    header_needed = not EVENTS_CSV.exists()
    with open(EVENTS_CSV, 'a', newline='') as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(['timestamp','object_id','direction','plate','conf','crop_path'])
        w.writerow([ts, oid, direction, plate or '', f"{conf:.3f}", str(crop_path)])

# ---------------- process event ----------------
def process_event_by_bbox(direction, bbox, frame):
    x,y,w,h = bbox
    pad_x = int(w * 0.6); pad_y = int(h * 0.8)
    x0 = max(0, x - pad_x); y0 = max(0, y - pad_y)
    x1 = min(frame.shape[1], x + w + pad_x); y1 = min(frame.shape[0], y + h + pad_y)
    crop = frame[y0:y1, x0:x1]
    tprint("[event] processing", direction, "crop_size", crop.shape)

    plate, conf = recognize_plate_local_improved(crop)
    tprint("[ANPR local] plate:", plate, "score:", conf)

    # save crop evidence
    ts = int(time.time())
    fn = OUTPUT_DIR / f"crop_oid{ts}_{direction}.jpg"
    try:
        cv2.imwrite(str(fn), crop)
    except Exception as e:
        tprint("cannot save crop:", e)
        fn = ""

    # accept if score >= threshold
    used_plate = plate if plate and conf >= ANPR_CONF_THRESHOLD else None

    # write event CSV: timestamp ISO, oid will be added by caller if known
    # For demo, use oid=None if not available
    append_event_csv(time.strftime("%Y-%m-%dT%H:%M:%S"), getattr(process_event_by_bbox, "_last_oid", None), direction, used_plate, conf, fn)
    tprint("[event] logged. plate used:", used_plate)

# ---------------- HTTP endpoint (manual) ----------------
@app.route("/event", methods=["POST"])
def http_event():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error":"invalid json"}), 400
    oid = data.get("object_id"); direction = data.get("direction")
    if oid is None or direction not in ("enter","exit"):
        return jsonify({"error":"invalid payload"}), 400
    with ct_lock:
        if oid not in tracker_bboxes:
            return jsonify({"error":"object_id not found"}), 404
        bbox = tracker_bboxes[oid]
        frame_snapshot = globals().get("latest_frame")
        if frame_snapshot is None:
            return jsonify({"error":"no frame available"}), 500
        # store oid for CSV entry
        process_event_by_bbox._last_oid = oid
        threading.Thread(target=process_event_by_bbox, args=(direction, bbox, frame_snapshot.copy()), daemon=True).start()
    return jsonify({"status":"accepted"}), 202

# ---------------- Detection + Tracking + Auto-crossing ----------------
def run_video_loop(video_path):
    global total_entered, total_exited

    if YOLO is None:
        tprint("ultralytics not installed. pip install ultralytics")
        sys.exit(1)

    tprint("Loading YOLOv8 model (yolov8n.pt) - may download weights")
    try:
        model = YOLO("yolov8n.pt")
    except Exception as e:
        tprint("Error loading YOLO:", e); sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        tprint("cannot open video:", video_path); sys.exit(1)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.2); continue
        frame_idx += 1
        if frame_idx % max(1, FRAME_SKIP) != 0:
            continue

        h, w = frame.shape[:2]
        try:
            ev = float(ENTRY_LINE)
            line_y = int(h*ev) if 0.0 < ev <= 1.0 else int(ev)
        except Exception:
            line_y = int(h*0.5)

        globals()["latest_frame"] = frame

        # inference
        try:
            results = model.predict(frame, conf=DETECT_CONF, iou=DETECT_IOU, device="cpu", verbose=False)
        except Exception as e:
            tprint("YOLO inference error:", e); time.sleep(0.2); continue

        preds = results[0]
        boxes = getattr(preds, "boxes", None)
        rects = []
        names = getattr(model, "names", {}) or {}

        if boxes is not None and len(boxes) > 0:
            try:
                xyxy = boxes.xyxy.cpu().numpy()
            except Exception:
                xyxy = np.array(boxes.xyxy)
            try:
                cls = boxes.cls.cpu().numpy().astype(int)
            except Exception:
                cls = np.array(boxes.cls).astype(int)
            for bb, c in zip(xyxy, cls):
                class_name = names.get(int(c), str(c)).lower()
                if class_name in VEHICLE_CLASSES:
                    x1,y1,x2,y2 = map(int, bb[:4])
                    wbox = x2-x1; hbox = y2-y1
                    if wbox <= 8 or hbox <= 8: continue
                    rects.append((x1,y1,wbox,hbox))

        # centroids
        new_centroids = []
        for (x,y,wbox,hbox) in rects:
            cx = int(x + wbox/2); cy = int(y + hbox/2)
            new_centroids.append((cx, cy, (x,y,wbox,hbox)))

        with ct_lock:
            matched_new = set()
            for idx, (cx,cy,bbox_xywh) in enumerate(new_centroids):
                for oid, cent in list(tracker_objects.items()):
                    dist = np.hypot(cent[0]-cx, cent[1]-cy)
                    if dist < MATCH_DISTANCE:
                        tracker_objects[oid] = (cx,cy)
                        tracker_bboxes[oid] = bbox_xywh
                        tracker_history.setdefault(oid, deque(maxlen=30)).append((cx,cy))
                        tracker_last_seen[oid] = time.time()
                        matched_new.add(idx)
                        break
            for idx, (cx,cy,bbox_xywh) in enumerate(new_centroids):
                if idx in matched_new: continue
                if len(tracker_objects) >= MAX_TRACKED_OBJECTS: continue
                oid = next_object_id_inc()
                tracker_objects[oid] = (cx,cy)
                tracker_bboxes[oid] = bbox_xywh
                tracker_history[oid] = deque(maxlen=30)
                tracker_history[oid].append((cx,cy))
                tracker_last_seen[oid] = time.time()
                tracker_counted_at[oid] = 0.0
                tprint("registered oid", oid, "centroid", (cx,cy))

            # crossing detection
            now = time.time()
            for oid, hist in list(tracker_history.items()):
                if len(hist) < 2: continue
                prev = hist[-2]; cur = hist[-1]
                prev_y = prev[1]; cur_y = cur[1]
                crossed = (prev_y < line_y <= cur_y) or (prev_y > line_y >= cur_y)
                if not crossed: continue
                last_count = tracker_counted_at.get(oid, 0.0)
                if now - last_count < COUNT_COOLDOWN: continue
                if prev_y < cur_y:
                    direction = "enter" if ENTRY_DIRECTION_IN == "down" else "exit"
                else:
                    direction = "exit" if ENTRY_DIRECTION_IN == "down" else "enter"
                with counters_lock:
                    if direction == "enter": total_entered += 1
                    else: total_exited += 1
                    tprint("auto-cross oid", oid, "dir", direction, "entered", total_entered, "exited", total_exited)
                tracker_counted_at[oid] = now
                # trigger ANPR processing
                bbox_use = tracker_bboxes.get(oid)
                frame_snapshot = globals().get("latest_frame")
                if bbox_use is not None and frame_snapshot is not None:
                    # tag last oid for CSV write
                    process_event_by_bbox._last_oid = oid
                    threading.Thread(target=process_event_by_bbox, args=(direction, bbox_use, frame_snapshot.copy()), daemon=True).start()

            # cleanup stale
            to_remove = [oid for oid, ts in tracker_last_seen.items() if now - ts > CLEANUP_SECONDS]
            for oid in to_remove:
                tracker_objects.pop(oid, None)
                tracker_bboxes.pop(oid, None)
                tracker_history.pop(oid, None)
                tracker_last_seen.pop(oid, None)
                tracker_counted_at.pop(oid, None)

        # debug overlay
        if CV_DEBUG:
            disp = frame.copy()
            cv2.line(disp, (0,line_y), (w,line_y), (0,0,255), 2)
            with ct_lock:
                for oid, (cx,cy) in tracker_objects.items():
                    bbox = tracker_bboxes.get(oid)
                    if bbox:
                        x,y,wb,hb = bbox
                        cv2.rectangle(disp, (x,y), (x+wb,y+hb), (0,255,0), 2)
                        cv2.putText(disp, f"ID:{oid}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(disp, f"Entered:{total_entered}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(disp, f"Exited:{total_exited}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("local_anpr", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if CV_DEBUG:
        cv2.destroyAllWindows()

# ---------------- CLI simulate ----------------
def cli_simulate():
    tprint("CLI simulate: l list, e <id> enter, x <id> exit, q quit")
    while True:
        cmd = input("> ").strip().split()
        if not cmd: continue
        if cmd[0] == "l":
            with ct_lock:
                tprint("objects", list(tracker_objects.keys()))
        elif cmd[0] == "e" and len(cmd)>1:
            oid = int(cmd[1])
            with ct_lock:
                if oid in tracker_bboxes and globals().get("latest_frame") is not None:
                    process_event_by_bbox._last_oid = oid
                    process_event_by_bbox("enter", tracker_bboxes[oid], globals()["latest_frame"].copy())
                else:
                    tprint("not found or no frame")
        elif cmd[0] == "x" and len(cmd)>1:
            oid = int(cmd[1])
            with ct_lock:
                if oid in tracker_bboxes and globals().get("latest_frame") is not None:
                    process_event_by_bbox._last_oid = oid
                    process_event_by_bbox("exit", tracker_bboxes[oid], globals()["latest_frame"].copy())
                else:
                    tprint("not found or no frame")
        elif cmd[0] == "q":
            break
        else:
            tprint("unknown command")

# ---------------- Flask starter ----------------
def start_flask_thread():
    def run():
        app.run(host="0.0.0.0", port=5001, debug=False)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=VIDEO_PATH)
    parser.add_argument("--cli", action="store_true")
    args = parser.parse_args()

    # init events CSV if not exist
    if not EVENTS_CSV.exists():
        with open(EVENTS_CSV, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['timestamp','object_id','direction','plate','conf','crop_path'])

    start_flask_thread()
    if args.cli:
        video_thread = threading.Thread(target=run_video_loop, args=(args.video,), daemon=True)
        video_thread.start()
        cli_simulate()
    else:
        try:
            run_video_loop(args.video)
        except KeyboardInterrupt:
            tprint("stopped by user")

if __name__ == "__main__":
    main()