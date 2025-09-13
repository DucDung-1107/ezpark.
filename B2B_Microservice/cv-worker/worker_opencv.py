#!/usr/bin/env python3
"""
worker_opencv.py
OpenCV ROI-based demo worker:
- đọc video (file mp4 hoặc RTSP)
- phát hiện chuyển động per-spot (ROI polygons) dùng BackgroundSubtractor MOG2 per ROI
- detect moving objects -> centroid tracker -> detect crossing a gate line
- when crossing in: allocate a free spot (from backend) and send slot_update occupied
- when crossing out: free earliest allocated spot and send slot_update free
- fallback: if no backend configured, print events

Cấu hình:
- spots.json (project root) định nghĩa site_id, list spots (id, polygon - list of [x,y] in pixel coords)
- biến môi trường:
    BACKEND_URL (ví dụ http://backend:8000)
    SITE_ID (site id)
    VIDEO_PATH (path to mp4 or rtsp url)
    FRAME_SKIP (process every Nth frame)
    MIN_CONTOUR_AREA (min area to consider)

-----------# Chạy:
export VIDEO_PATH="/Users/hoangquannguyen/Downloads/oto.mov"
export BACKEND_URL="http://localhost:8000"  # nếu backend local
export SITE_ID="site_demo_1"
python cv-worker/worker_opencv.py
"""

import os
import sys
import time
import json
import math
import argparse
from collections import deque
from dotenv import load_dotenv

load_dotenv()

import cv2
import numpy as np
import requests
from typing import List, Tuple, Dict

# Config from env
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")
SITE_ID = os.environ.get("SITE_ID", "site_demo_1")
VIDEO_PATH = os.environ.get("VIDEO_PATH", "/video/input.mp4")  # mount into container
FRAME_SKIP = int(os.environ.get("FRAME_SKIP", "2"))
MIN_CONTOUR_AREA = int(os.environ.get("MIN_CONTOUR_AREA", "400"))
DEBUG = os.environ.get("CV_DEBUG", "1") == "1"
ASSIGN_STRATEGY = os.environ.get("ASSIGN_STRATEGY", "fifo")  # fifo or stack


def load_spots_config(candidate_paths=None):
    if candidate_paths is None:
        candidate_paths = [
            "/spots.json",
            "/usr/src/app/spots.json",
            "./spots.json",
        ]
    for p in candidate_paths:
        try:
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                print("[worker] Loaded spots.json from:", p)
                return doc
        except Exception as e:
            print("[worker] Error loading", p, ":", e)
    print("[worker] No spots.json found. Exiting.")
    return None


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
        print("[worker] POST slot_update", spot_id, status, "->", r.status_code)
        return r
    except Exception as e:
        print("[worker] Error POST slot_update:", e)
        return None


# Simple centroid tracker (small, robust)
class CentroidTracker:
    def __init__(self, maxDisappeared=10, maxDistance=50):
        # next object ID
        self.nextObjectID = 0
        # objectID -> centroid tuple
        self.objects = dict()
        # objectID -> disappeared frames count
        self.disappeared = dict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance
        # store history of centroids
        self.history = dict()

    def register(self, centroid):
        oid = self.nextObjectID
        self.objects[oid] = centroid
        self.disappeared[oid] = 0
        self.history[oid] = deque(maxlen=30)
        self.history[oid].append(centroid)
        self.nextObjectID += 1
        return oid

    def deregister(self, objectID):
        if objectID in self.objects:
            del self.objects[objectID]
        if objectID in self.disappeared:
            del self.disappeared[objectID]
        if objectID in self.history:
            del self.history[objectID]

    def update(self, rects: List[Tuple[int, int, int, int]]):
        # rects: list of bounding boxes (x, y, w, h)
        if len(rects) == 0:
            # mark all as disappeared
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    self.deregister(oid)
            return self.objects

        # compute centroids
        inputCentroids = []
        for (x, y, w, h) in rects:
            cX = int(x + w / 2.0)
            cY = int(y + h / 2.0)
            inputCentroids.append((cX, cY))

        # if no existing objects, register all
        if len(self.objects) == 0:
            for c in inputCentroids:
                oid = self.register(c)
        else:
            # match existing object centroids to inputCentroids via Euclidean
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = np.linalg.norm(np.array(objectCentroids)[:, None] - np.array(inputCentroids)[None, :], axis=2)
            # find smallest distances
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > self.maxDistance:
                    continue
                oid = objectIDs[row]
                self.objects[oid] = inputCentroids[col]
                self.history[oid].append(inputCentroids[col])
                self.disappeared[oid] = 0
                usedRows.add(row)
                usedCols.add(col)

            # register new inputCentroids that were not matched
            for col in range(len(inputCentroids)):
                if col not in usedCols:
                    self.register(inputCentroids[col])

            # increase disappeared for unmatched objectIDs
            for row in range(len(objectCentroids)):
                if row not in usedRows:
                    oid = objectIDs[row]
                    self.disappeared[oid] += 1
                    if self.disappeared[oid] > self.maxDisappeared:
                        self.deregister(oid)
        return self.objects


# Util: check if line segment AB intersects segment CD
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def point_side(pt, line_p1, line_p2):
    # return positive/negative side
    x, y = pt
    x1, y1 = line_p1
    x2, y2 = line_p2
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def bbox_area(bbox):
    _, _, w, h = bbox
    return w * h


def create_mask_from_polygon(poly, frame_shape):
    mask = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
    pts = np.array(poly, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def main():
    cfg = load_spots_config()
    if not cfg:
        sys.exit(1)

    site_id = cfg.get("site_id", SITE_ID)
    spots_cfg = cfg.get("spots", [])
    gate_cfg = cfg.get("gate_line", None)  # ex: [[x1,y1],[x2,y2]]
    capacity = len(spots_cfg)

    print(f"[worker] site={site_id} spots={capacity} gate={gate_cfg}")

    # init per-spot background subtractors and masks
    spot_subtractors = {}
    spot_masks = {}
    spot_bboxes = {}  # bounding rect per polygon
    spot_last_status = {}
    spot_status_stable_count = {}
    STABLE_THRESHOLD = 3  # need N consecutive frames to change state

    # compute polygon masks later when we know frame size
    spots_polys = []
    for s in spots_cfg:
        poly = s.get("polygon") or s.get("poly")  # polygon: [[x,y],...]
        spots_polys.append((s["id"], s.get("code", s["id"]), poly))

    # centroid tracker & allocated spots queue
    tracker = CentroidTracker(maxDisappeared=15, maxDistance=80)
    allocated_spots = deque()  # FIFO list of spot ids assigned to entries

    # For fallback occupancy count using allocations
    current_allocated = set()

    # open video
    print("[worker] opening video:", VIDEO_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[worker] ERROR: cannot open video:", VIDEO_PATH)
        sys.exit(1)

    # read first frame to get size
    ret, frame = cap.read()
    if not ret:
        print("[worker] ERROR: cannot read first frame")
        sys.exit(1)
    frame_h, frame_w = frame.shape[:2]
    print(f"[worker] frame size: {frame_w}x{frame_h}")

    # prepare spot masks, subtractors
    for (sid, code, poly) in spots_polys:
        if not poly:
            continue
        # if polygon coordinates are normalized (0..1), scale to pixel coords
        try:
            # check first point
            if 0 < poly[0][0] <= 1 and 0 < poly[0][1] <= 1:
                scaled = [[int(x * frame_w), int(y * frame_h)] for (x, y) in poly]
            else:
                scaled = [[int(x), int(y)] for (x, y) in poly]
        except Exception:
            scaled = poly
        mask = create_mask_from_polygon(scaled, frame.shape)
        x, y, w, h = cv2.boundingRect(np.array(scaled, dtype=np.int32))
        spot_masks[sid] = (mask, (x, y, w, h))
        spot_subtractors[sid] = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)
        spot_last_status[sid] = "unknown"
        spot_status_stable_count[sid] = 0
        spot_bboxes[sid] = (x, y, w, h)
        print(f"[worker] spot {sid} bbox={x,y,w,h}")

    # gate line setup
    if gate_cfg and isinstance(gate_cfg, list) and len(gate_cfg) == 2:
        # scale gate coords if normalized
        g0, g1 = gate_cfg
        if 0 < g0[0] <= 1 and 0 < g0[1] <= 1:
            gate_p1 = (int(g0[0] * frame_w), int(g0[1] * frame_h))
            gate_p2 = (int(g1[0] * frame_w), int(g1[1] * frame_h))
        else:
            gate_p1 = (int(g0[0]), int(g0[1]))
            gate_p2 = (int(g1[0]), int(g1[1]))
        print("[worker] gate line:", gate_p1, gate_p2)
    else:
        # default: horizontal line near bottom (vehicles moving up are exit; moving down are entry)
        gate_p1 = (int(frame_w * 0.1), int(frame_h * 0.8))
        gate_p2 = (int(frame_w * 0.9), int(frame_h * 0.8))
        print("[worker] no gate_line in config, using default:", gate_p1, gate_p2)

    # processing loop
    frame_i = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[worker] video ended, looping back to start")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.5)
                continue

            frame_i += 1
            if frame_i % FRAME_SKIP != 0:
                continue

            # resize to reasonable size for speed (optional)
            # frame = cv2.resize(frame, (frame_w, frame_h))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # === per-spot detection ===
            for sid, (mask, bbox) in spot_masks.items():
                x, y, w, h = bbox
                # crop ROI
                roi = frame[y : y + h, x : x + w]
                if roi.size == 0:
                    continue
                fgmask = spot_subtractors[sid].apply(roi)
                # mask out non-ROI area inside bbox using mask slice
                mask_crop = mask[y : y + h, x : x + w]
                if mask_crop.shape != fgmask.shape:
                    # adapt mask if weird shapes
                    mask_crop = cv2.resize(mask_crop, (fgmask.shape[1], fgmask.shape[0]))
                fg = cv2.bitwise_and(fgmask, fgmask, mask=mask_crop)
                # threshold and count non-zero
                _, th = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
                nz = cv2.countNonZero(th)
                area = w * h
                ratio = nz / (area + 1e-6)
                occupied = ratio > 0.02  # heuristic; tune if needed
                new_status = "occupied" if occupied else "free"
                if new_status != spot_last_status[sid]:
                    spot_status_stable_count[sid] += 1
                else:
                    spot_status_stable_count[sid] = 0
                # only accept change if stable for STABLE_THRESHOLD frames
                if spot_status_stable_count[sid] >= STABLE_THRESHOLD:
                    if new_status != spot_last_status[sid]:
                        # change state
                        spot_last_status[sid] = new_status
                        # send event to backend
                        print(f"[worker] spot {sid} state -> {new_status} (ratio={ratio:.4f})")
                        post_slot_update(site_id, sid, new_status)
                # debug overlay
                if DEBUG:
                    color = (0, 255, 0) if spot_last_status[sid] == "free" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{sid}:{spot_last_status[sid]}", (x + 2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # === global motion detection for gate tracking ===
            # simple overall background subtraction for moving objects
            # create global subtractor lazily
            if "global_sub" not in locals():
                global_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=True)
            fgmask = global_sub.apply(frame)
            _, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rects = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_CONTOUR_AREA:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                rects.append((x, y, w, h))

            objects = tracker.update(rects)

            # check crossing events
            for oid, centroid in list(tracker.objects.items()):
                hist = tracker.history.get(oid, [])
                if len(hist) < 2:
                    continue
                prev = hist[-2]
                curr = hist[-1]
                # check intersection of motion segment with gate line
                if intersect(prev, curr, gate_p1, gate_p2):
                    # determine direction by side sign before/after
                    side_prev = point_side(prev, gate_p1, gate_p2)
                    side_curr = point_side(curr, gate_p1, gate_p2)
                    direction = None
                    if side_prev < 0 and side_curr > 0:
                        direction = "enter"
                    elif side_prev > 0 and side_curr < 0:
                        direction = "exit"
                    else:
                        # unknown direction; approximate using y movement
                        dy = curr[1] - prev[1]
                        direction = "enter" if dy > 0 else "exit"
                    print(f"[worker] object {oid} crossed gate: {direction}")
                    # take action: allocate or free spot
                    if direction == "enter":
                        # find first free spot from backend, else local heuristic
                        spot_id = None
                        try:
                            r = requests.get(f"{BACKEND_URL}/api/spots?site_id={site_id}", timeout=4)
                            if r.status_code == 200:
                                spots = r.json().get("spots", [])
                                for s in spots:
                                    if s.get("status") in (None, "", "free"):
                                        spot_id = s["id"]
                                        break
                        except Exception as e:
                            print("[worker] error fetching spots from backend:", e)
                        if not spot_id:
                            # fallback: choose any spot id not already allocated
                            for (s, _, _) in spots_polys:
                                if s not in current_allocated:
                                    spot_id = s
                                    break
                        if spot_id:
                            allocated_spots.append(spot_id)
                            current_allocated.add(spot_id)
                            print(f"[worker] allocated spot {spot_id} for entry")
                            post_slot_update(site_id, spot_id, "occupied")
                        else:
                            print("[worker] no free spot to allocate (entry event)")
                    else:  # exit
                        # free earliest allocated spot (FIFO)
                        if allocated_spots:
                            sid = allocated_spots.popleft()
                            if sid in current_allocated:
                                current_allocated.remove(sid)
                            print(f"[worker] freeing allocated spot {sid} for exit")
                            post_slot_update(site_id, sid, "free")
                        else:
                            # fallback: try query backend for an occupied spot to free
                            try:
                                r = requests.get(f"{BACKEND_URL}/api/spots?site_id={site_id}", timeout=4)
                                if r.status_code == 200:
                                    spots = r.json().get("spots", [])
                                    occ = [s for s in spots if s.get("status") == "occupied"]
                                    if occ:
                                        sid = occ[0]["id"]
                                        print(f"[worker] (fallback) freeing backend occupied spot {sid}")
                                        post_slot_update(site_id, sid, "free")
                            except Exception as e:
                                print("[worker] error fallback free:", e)

            # debug overlay: draw gate line
            if DEBUG:
                cv2.line(frame, gate_p1, gate_p2, (255, 0, 0), 2)
                cv2.putText(frame, f"Allocated: {len(current_allocated)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                # show frame
                cv2.imshow("worker_debug", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("[worker] interrupted by user")
    finally:
        cap.release()
        if DEBUG:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()