#!/usr/bin/env python3
"""
worker_no_gate.py

Mô tả:
- Đọc video (file .mp4 hoặc RTSP via VIDEO_PATH)
- Dò chuyển động / detect contours -> duy trì simple centroid tracker (object IDs)
- Expose HTTP endpoint POST /event to trigger checkin/checkout:
    POST /event  JSON: {"object_id": <int>, "direction": "enter" | "exit"}
- Khi event nhận: crop vùng quanh bbox của object -> gọi ANPR (cloud PlateRecognizer nếu có API key),
  rồi gọi backend để check-in / check-out (POST /api/events/slot_update).
- Fallback: nếu ANPR không đọc được plate thì dùng FIFO allocation/freeing.

Env vars (.env):
- VIDEO_PATH (default /video/input.mp4)
- BACKEND_URL (default http://backend:8000)
- SITE_ID (default site_demo_1)
- PLATE_RECOGNIZER_API_KEY (optional)
- PLATE_RECOGNIZER_URL (default PlateRecognizer endpoint)
- CV_DEBUG (1 để show window; 0 để headless)
- FRAME_SKIP, MIN_CONTOUR_AREA

------------# Chạy:
pip install -r cv-worker/requirements.txt
export VIDEO_PATH="/Users/hoangquannguyen/Downloads/oto.mov"
export BACKEND_URL="http://localhost:8000"   
export SITE_ID="site_demo_1"
export PLATE_RECOGNIZER_API_KEY="your_key"   
export CV_DEBUG=1                            
python cv-worker/worker_no_gate.py --video "$VIDEO_PATH"

"""

import os
import sys
import time
import json
import io
import argparse
import threading
from collections import deque

from dotenv import load_dotenv
load_dotenv()

import cv2
import numpy as np
import requests
from PIL import Image
from flask import Flask, request, jsonify

# -------------------- Config (env-friendly) --------------------
VIDEO_PATH = os.environ.get("VIDEO_PATH", "/video/input.mp4")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")
SITE_ID = os.environ.get("SITE_ID", "site_demo_1")
PLATE_API_KEY = os.environ.get("PLATE_RECOGNIZER_API_KEY", "")
PLATE_API_URL = os.environ.get("PLATE_RECOGNIZER_URL", "https://api.platerecognizer.com/v1/plate-reader/")
FRAME_SKIP = int(os.environ.get("FRAME_SKIP", "2"))
MIN_CONTOUR_AREA_DEFAULT = int(os.environ.get("MIN_CONTOUR_AREA", "2000"))
MAX_BOXES_DEFAULT = int(os.environ.get("MAX_BOXES", "12"))
ASPECT_RATIO_MIN_DEFAULT = float(os.environ.get("ASPECT_RATIO_MIN", "0.6"))
ASPECT_RATIO_MAX_DEFAULT = float(os.environ.get("ASPECT_RATIO_MAX", "4.5"))
SOLIDITY_MIN_DEFAULT = float(os.environ.get("SOLIDITY_MIN", "0.25"))
NMS_IOU_DEFAULT = float(os.environ.get("NMS_IOU", "0.35"))
CLEANUP_SECONDS_DEFAULT = float(os.environ.get("CLEANUP_SECONDS", "2.5"))
MAX_TRACKED_OBJECTS_DEFAULT = int(os.environ.get("MAX_TRACKED_OBJECTS", "80"))
MIN_CONTOUR_AREA = MIN_CONTOUR_AREA_DEFAULT
MAX_BOXES = MAX_BOXES_DEFAULT
ASPECT_RATIO_MIN = ASPECT_RATIO_MIN_DEFAULT
ASPECT_RATIO_MAX = ASPECT_RATIO_MAX_DEFAULT
SOLIDITY_MIN = SOLIDITY_MIN_DEFAULT
NMS_IOU = NMS_IOU_DEFAULT
CLEANUP_SECONDS = CLEANUP_SECONDS_DEFAULT
MAX_TRACKED_OBJECTS = MAX_TRACKED_OBJECTS_DEFAULT

CV_DEBUG = os.environ.get("CV_DEBUG", "0") == "1"

# -------------------- Internal global state --------------------
tracker_objects = {}      # objectID -> centroid (x,y)
tracker_bboxes = {}       # objectID -> bbox (x,y,w,h)
tracker_history = {}      # objectID -> deque of centroids
tracker_last_seen = {}    # objectID -> last seen timestamp
next_object_id = 0
ct_lock = threading.Lock()

allocated_spots = deque()  # FIFO allocations by this worker
current_allocated = set()  # set of spot ids we think allocated

# For exposing API
app = Flask(__name__)

# -------------------- ANPR helper --------------------
def recognize_plate_cloud(image_bgr):
    """
    Call a cloud ANPR service (PlateRecognizer style). Return (plate_text, confidence) or (None, 0.0).
    """
    if not PLATE_API_KEY:
        return None, 0.0
    try:
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG")
        jbytes = buffer.getvalue()
        headers = {"Authorization": f"Token {PLATE_API_KEY}"}
        files = {"upload": ("frame.jpg", jbytes, "image/jpeg")}
        r = requests.post(PLATE_API_URL, headers=headers, files=files, timeout=10)
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", []) or data.get("data", [])
            if results:
                first = results[0]
                plate = first.get("plate") or first.get("vehicle_plate") or first.get("plate_number") or ""
                conf = first.get("score") or first.get("confidence") or 0.0
                return plate.strip(), float(conf)
    except Exception as e:
        print("[ANPR] cloud error:", e)
    return None, 0.0

# -------------------- Backend interactions --------------------
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
        print("[worker] BACKEND_URL not set, would send:", payload)
        return None
    url = f"{BACKEND_URL}/api/events/slot_update"
    try:
        r = requests.post(url, json=payload, timeout=6)
        print("[worker] POST slot_update", spot_id, status, "plate:", plate, "->", r.status_code)
        return r
    except Exception as e:
        print("[worker] Error POST slot_update:", e)
        return None

def find_free_spot_via_backend(site_id):
    try:
        r = requests.get(f"{BACKEND_URL}/api/spots?site_id={site_id}", timeout=4)
        if r.status_code == 200:
            for s in r.json().get("spots", []):
                if s.get("status") in (None, "", "free"):
                    return s["id"]
    except Exception as e:
        print("[worker] error fetching spots:", e)
    return None

def find_session_by_plate(plate):
    try:
        r = requests.get(f"{BACKEND_URL}/api/sessions?plate={plate}", timeout=5)
        if r.status_code == 200:
            sessions = r.json().get("sessions", [])
            for s in sessions:
                if s.get("status") == "active":
                    return s
    except Exception as e:
        print("[worker] error fetching sessions by plate:", e)
    return None

# -------------------- Event processing --------------------
def process_event_by_bbox(direction, bbox, frame):
    """
    direction: "enter" or "exit"
    bbox: (x,y,w,h)
    frame: BGR numpy array
    """
    x, y, w, h = bbox
    # expand crop a bit (configurable heuristics)
    pad_x = int(w * 0.6)
    pad_y = int(h * 0.8)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(frame.shape[1], x + w + pad_x)
    y1 = min(frame.shape[0], y + h + pad_y)
    crop = frame[y0:y1, x0:x1]
    print(f"[worker] process_event {direction} crop size: {crop.shape}")

    # ANPR (cloud)
    plate, conf = recognize_plate_cloud(crop)
    print(f"[worker] ANPR result: plate={plate} conf={conf}")

    if direction == "enter":
        if plate:
            spot_id = find_free_spot_via_backend(SITE_ID)
            if spot_id:
                allocated_spots.append(spot_id)
                current_allocated.add(spot_id)
                post_slot_update(SITE_ID, spot_id, "occupied", image_url=None, plate=plate)
            else:
                print("[worker] enter: no free spot found via backend")
        else:
            # fallback: allocate first free via backend (without plate)
            spot_id = find_free_spot_via_backend(SITE_ID)
            if spot_id:
                allocated_spots.append(spot_id)
                current_allocated.add(spot_id)
                post_slot_update(SITE_ID, spot_id, "occupied", image_url=None, plate=None)
            else:
                print("[worker] enter: cannot allocate (no plate, no free spot)")

    elif direction == "exit":
        if plate:
            sess = find_session_by_plate(plate)
            if sess:
                spot_id = sess.get("spot_id")
                if spot_id:
                    post_slot_update(SITE_ID, spot_id, "free", image_url=None, plate=plate)
                    try:
                        allocated_spots.remove(spot_id)
                        current_allocated.discard(spot_id)
                    except Exception:
                        pass
                else:
                    # fallback: free first allocated
                    if allocated_spots:
                        sid = allocated_spots.popleft()
                        current_allocated.discard(sid)
                        post_slot_update(SITE_ID, sid, "free", image_url=None, plate=plate)
                    else:
                        print("[worker] exit: session has no spot_id; no allocated to free")
            else:
                # no session found -> fallback free first allocated
                if allocated_spots:
                    sid = allocated_spots.popleft()
                    current_allocated.discard(sid)
                    post_slot_update(SITE_ID, sid, "free", image_url=None, plate=plate)
                else:
                    # fallback free any backend occupied
                    try:
                        r = requests.get(f"{BACKEND_URL}/api/spots?site_id={SITE_ID}", timeout=4)
                        if r.status_code == 200:
                            spots = r.json().get("spots", [])
                            for s in spots:
                                if s.get("status") == "occupied":
                                    post_slot_update(SITE_ID, s["id"], "free", image_url=None, plate=plate)
                                    break
                    except Exception as e:
                        print("[worker] exit fallback error:", e)
        else:
            # no plate: free FIFO
            if allocated_spots:
                sid = allocated_spots.popleft()
                current_allocated.discard(sid)
                post_slot_update(SITE_ID, sid, "free", image_url=None, plate=None)
            else:
                print("[worker] exit: no plate and no allocated spots")

# -------------------- HTTP endpoint --------------------
@app.route("/event", methods=["POST"])
def http_event():
    """
    Expect JSON: {"object_id": <int>, "direction": "enter"|"exit"}
    """
    data = request.get_json(force=True)
    oid = data.get("object_id")
    direction = data.get("direction")
    if oid is None or direction not in ("enter", "exit"):
        return jsonify({"error": "invalid payload"}), 400

    with ct_lock:
        if oid not in tracker_bboxes:
            return jsonify({"error": "object_id not found"}), 404
        bbox = tracker_bboxes[oid]
        frame_snapshot = globals().get("latest_frame")
        if frame_snapshot is None:
            return jsonify({"error": "no frame available"}), 500
        # process in background thread
        threading.Thread(target=process_event_by_bbox, args=(direction, bbox, frame_snapshot.copy()), daemon=True).start()
    return jsonify({"status": "accepted", "object_id": oid, "direction": direction}), 202

# -------------------- Video processing loop (with improved filtering and cleanup) --------------------
def run_video_loop(video_path):
    global next_object_id, tracker_objects, tracker_bboxes, tracker_history, tracker_last_seen

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[worker] cannot open video:", video_path)
        sys.exit(1)

    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=True)

    frame_i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            # loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.2)
            continue

        frame_i += 1
        if frame_i % FRAME_SKIP != 0:
            continue

        # publish latest frame for HTTP handler
        globals()["latest_frame"] = frame

        # background subtraction and threshold
        fg = backSub.apply(frame)
        _, th = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        # morphological operations to merge fragments: close then dilate
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        th = cv2.dilate(th, kernel_close, iterations=1)

        # find contours
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # dynamic params (allow runtime tuning via env)
        MIN_CONTOUR_AREA = int(os.environ.get("MIN_CONTOUR_AREA", MIN_CONTOUR_AREA_DEFAULT))
        MAX_BOXES = int(os.environ.get("MAX_BOXES", MAX_BOXES_DEFAULT))
        ASPECT_RATIO_MIN = float(os.environ.get("ASPECT_RATIO_MIN", ASPECT_RATIO_MIN_DEFAULT))
        ASPECT_RATIO_MAX = float(os.environ.get("ASPECT_RATIO_MAX", ASPECT_RATIO_MAX_DEFAULT))
        SOLIDITY_MIN = float(os.environ.get("SOLIDITY_MIN", SOLIDITY_MIN_DEFAULT))
        NMS_IOU = float(os.environ.get("NMS_IOU", NMS_IOU_DEFAULT))

        rects_raw = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # aspect ratio filter
            ar = w / float(h + 1e-6)
            if ar < ASPECT_RATIO_MIN or ar > ASPECT_RATIO_MAX:
                continue
            bbox_area = w * h
            solidity = area / float(bbox_area + 1e-6)
            if solidity < SOLIDITY_MIN:
                continue
            rects_raw.append((x, y, w, h, area))

        # if none, set empty rects
        if not rects_raw:
            rects = []
        else:
            # keep top-N by area
            rects_raw = sorted(rects_raw, key=lambda r: r[4], reverse=True)[:MAX_BOXES]

            # simple NMS to remove overlapping boxes
            def iou_rect(a, b):
                ax, ay, aw, ah = a
                bx, by, bw, bh = b
                x1 = max(ax, bx)
                y1 = max(ay, by)
                x2 = min(ax + aw, bx + bw)
                y2 = min(ay + ah, by + bh)
                inter_w = max(0, x2 - x1)
                inter_h = max(0, y2 - y1)
                inter = inter_w * inter_h
                union = aw * ah + bw * bh - inter
                return 0.0 if union <= 0 else inter / union

            picked = []
            for box in rects_raw:
                x, y, w, h, area = box
                keep = True
                for p in picked:
                    if iou_rect((x, y, w, h), (p[0], p[1], p[2], p[3])) > NMS_IOU:
                        keep = False
                        break
                if keep:
                    picked.append((x, y, w, h, area))
            rects = [(p[0], p[1], p[2], p[3]) for p in picked]

        # prepare centroids for matching
        new_centroids = []
        for (x, y, w, h) in rects:
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            new_centroids.append((cx, cy, (x, y, w, h)))

        with ct_lock:
            # matching: naive nearest centroid matching
            used_oids = set()
            matched_new = set()
            # match existing tracker objects first
            for idx, (cx, cy, bbox_xywh) in enumerate(new_centroids):
                matched = False
                # search for nearest existing centroid
                for oid, cent in list(tracker_objects.items()):
                    dist = np.hypot(cent[0] - cx, cent[1] - cy)
                    if dist < 60:  # match threshold
                        tracker_objects[oid] = (cx, cy)
                        tracker_bboxes[oid] = bbox_xywh
                        tracker_history.setdefault(oid, deque(maxlen=30)).append((cx, cy))
                        tracker_last_seen[oid] = time.time()
                        used_oids.add(oid)
                        matched = True
                        matched_new.add(idx)
                        break
                # if not matched, will register after checking all existing
            # register new centroids that were not matched
            for idx, (cx, cy, bbox_xywh) in enumerate(new_centroids):
                if idx in matched_new:
                    continue
                # limit total tracked objects
                if len(tracker_objects) >= int(os.environ.get("MAX_TRACKED_OBJECTS", MAX_TRACKED_OBJECTS_DEFAULT)):
                    # skip registering new tiny ones if tracking cap reached
                    continue
                oid = next_object_id_local_increment()
                tracker_objects[oid] = (cx, cy)
                tracker_bboxes[oid] = bbox_xywh
                tracker_history[oid] = deque(maxlen=30)
                tracker_history[oid].append((cx, cy))
                tracker_last_seen[oid] = time.time()
                print(f"[worker] registered object {oid} at {(cx, cy)}")

            # cleanup disappeared objects
            current_time = time.time()
            CLEANUP_SECONDS = float(os.environ.get("CLEANUP_SECONDS", CLEANUP_SECONDS_DEFAULT))
            to_remove = []
            for oid, last in list(tracker_last_seen.items()):
                if current_time - last > CLEANUP_SECONDS:
                    to_remove.append(oid)
            for oid in to_remove:
                tracker_objects.pop(oid, None)
                tracker_bboxes.pop(oid, None)
                tracker_history.pop(oid, None)
                tracker_last_seen.pop(oid, None)

            # enforce max tracked objects
            MAX_TRACKED_OBJECTS = int(os.environ.get("MAX_TRACKED_OBJECTS", MAX_TRACKED_OBJECTS_DEFAULT))
            if len(tracker_objects) > MAX_TRACKED_OBJECTS:
                # remove oldest by last_seen
                items = sorted(tracker_last_seen.items(), key=lambda kv: kv[1])
                remove_count = len(tracker_objects) - MAX_TRACKED_OBJECTS
                for oid, _ in items[:remove_count]:
                    tracker_objects.pop(oid, None)
                    tracker_bboxes.pop(oid, None)
                    tracker_history.pop(oid, None)
                    tracker_last_seen.pop(oid, None)

        # debug display
        if CV_DEBUG:
            disp = frame.copy()
            with ct_lock:
                for oid, (cx, cy) in tracker_objects.items():
                    bbox = tracker_bboxes.get(oid)
                    if bbox:
                        x, y, w, h = bbox
                        cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(disp, f"ID:{oid}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("worker_no_gate", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if CV_DEBUG:
        cv2.destroyAllWindows()

# -------------------- helper to safely increment global next_object_id --------------------
_next_object_id_lock = threading.Lock()
def next_object_id_local_increment():
    global next_object_id
    with _next_object_id_lock:
        oid = next_object_id
        next_object_id += 1
    return oid

# -------------------- CLI helper --------------------
def cli_simulate():
    print("CLI simulate mode: 'l' list, 'e <id>' enter, 'x <id>' exit, 'q' quit")
    while True:
        cmd = input("> ").strip().split()
        if not cmd:
            continue
        if cmd[0] == "l":
            with ct_lock:
                print("objects:", list(tracker_objects.keys()))
                for oid in tracker_objects.keys():
                    print(oid, "bbox=", tracker_bboxes.get(oid), "centroid=", tracker_objects.get(oid))
        elif cmd[0] == "e" and len(cmd) > 1:
            try:
                oid = int(cmd[1])
                with ct_lock:
                    if oid in tracker_bboxes and globals().get("latest_frame") is not None:
                        process_event_by_bbox("enter", tracker_bboxes[oid], globals()["latest_frame"].copy())
                    else:
                        print("object not found or no frame")
            except Exception as e:
                print("err:", e)
        elif cmd[0] == "x" and len(cmd) > 1:
            try:
                oid = int(cmd[1])
                with ct_lock:
                    if oid in tracker_bboxes and globals().get("latest_frame") is not None:
                        process_event_by_bbox("exit", tracker_bboxes[oid], globals()["latest_frame"].copy())
                    else:
                        print("object not found or no frame")
            except Exception as e:
                print("err:", e)
        elif cmd[0] == "q":
            print("exit CLI")
            break
        else:
            print("unknown command")

# -------------------- Web server thread --------------------
def start_flask_thread():
    def run():
        app.run(host="0.0.0.0", port=5001, debug=False)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=VIDEO_PATH, help="video path")
    parser.add_argument("--cli", action="store_true", help="enable CLI simulate mode")
    args = parser.parse_args()

    start_flask_thread()

    if args.cli:
        # start video processing in a background thread and then CLI in main thread
        video_thread = threading.Thread(target=run_video_loop, args=(args.video,), daemon=True)
        video_thread.start()
        cli_simulate()
    else:
        try:
            run_video_loop(args.video)
        except KeyboardInterrupt:
            print("stopped by user")

if __name__ == "__main__":
    main()