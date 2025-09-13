"""
python anpr_improved.py --video /Users/hoangquannguyen/Downloads/xemay_sang.mov
"""

import os, sys, time, argparse, csv, math, tempfile, uuid
from pathlib import Path
from collections import deque

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
try:
    import easyocr
except Exception:
    easyocr = None
try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None
try:
    import pytesseract
except Exception:
    pytesseract = None

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./anpr_out")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_CSV = OUTPUT_DIR / "events.csv"

VIDEO_SRC_DEFAULT = os.environ.get("VIDEO_SRC", "0")  
PLATE_YOLO_WEIGHTS = os.environ.get("PLATE_YOLO_WEIGHTS", "")  

FRAME_SKIP = int(os.environ.get("FRAME_SKIP", "2"))
DETECT_CONF = float(os.environ.get("DETECT_CONF", "0.35"))
DETECT_IOU = float(os.environ.get("DETECT_IOU", "0.45"))
MIN_PLATE_AREA_FRAC = float(os.environ.get("MIN_PLATE_AREA_FRAC", "0.0006"))
PLATE_ASPECT_MIN = float(os.environ.get("PLATE_ASPECT_MIN", "2.0"))
PLATE_ASPECT_MAX = float(os.environ.get("PLATE_ASPECT_MAX", "6.0"))

# scoring / uniqueness
OCR_SCORE_THRESHOLD = float(os.environ.get("OCR_SCORE_THRESHOLD", "0.4"))
PLATE_SEEN_COOLDOWN = float(os.environ.get("PLATE_SEEN_COOLDOWN", "35.0"))  

# debug
CV_DEBUG = os.environ.get("CV_DEBUG", "1") == "1"
SAVE_VARIANTS = os.environ.get("SAVE_VARIANTS", "1") == "1"

# movement / counting (simple: count when a new plate appears — user requested "không cần detect xe")
# We'll count once per new plate read (unique by cooldown)
seen_plates = {}  # plate -> last_seen_ts
count_entered = 0

# OCR backends lazy init
_paddle_reader = None
_easyocr_reader = None

# simple VN plate regex heuristic (adjust if another country)
import re
_VN_PLATE_REGEX = re.compile(r'([0-9]{2,3})\s*[-]?\s*([A-Z0-9]{1,4})\s*[-]?\s*([0-9]{3,4})', re.I)

# ------------------ Utils ------------------
def log(*args, **kwargs):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), *args, **kwargs)

def normalize_plate_text(s):
    if not s: return s
    s2 = re.sub(r'[^A-Za-z0-9\.\-]', '', s).upper()
    s2 = re.sub(r'\.+', '.', s2)
    return s2

def plate_regex_score(text):
    if not text: return 0.0
    t = normalize_plate_text(text)
    if _VN_PLATE_REGEX.search(t):
        return 1.0
    # heuristic: more alnum chars => slightly higher
    al = len(re.sub(r'[^A-Za-z0-9]', '', t))
    return min(0.6, 0.08 * al)

def ensure_paddle():
    global _paddle_reader
    if PaddleOCR is None:
        return False
    if _paddle_reader is None:
        # English model good for alphanumeric; set use_angle_cls=False for speed
        _paddle_reader = PaddleOCR(use_angle_cls=False, lang='en')
    return True

def ensure_easyocr():
    global _easyocr_reader
    if easyocr is None:
        return False
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(['en'], gpu=False)
    return True

def run_paddleocr(img_bgr):
    if not ensure_paddle():
        return []
    # paddle expects numpy BGR (or accept path). We'll pass image.
    try:
        res = _paddle_reader.ocr(img_bgr, cls=False)
        outputs = []
        for line in res:
            for seg in line:
                txt = seg[1][0]
                conf = float(seg[1][1]) if seg[1][1] else 0.0
                outputs.append((txt, conf))
        return outputs
    except Exception as e:
        log("PaddleOCR error:", e)
        return []

def run_easyocr(img_bgr):
    if not ensure_easyocr():
        return []
    try:
        res = _easyocr_reader.readtext(img_bgr)
        return [(txt, float(conf)) for (_, txt, conf) in res]
    except Exception as e:
        log("EasyOCR error:", e)
        return []

def run_tesseract(img_bgr):
    if pytesseract is None:
        return []
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        txt = pytesseract.image_to_string(gray, config='--psm 7')  # line mode
        txt = txt.strip()
        return [(txt, 0.45 if txt else 0.0)]
    except Exception as e:
        log("tesseract error:", e)
        return []

# ------------------ Plate detection helpers ------------------
def detect_plates_with_yolo(frame, model):
    """
    Use YOLOv8 plate-specific model to detect plates.
    Returns list of bbox tuples (x,y,w,h,score)
    """
    results = model.predict(frame, conf=DETECT_CONF, iou=DETECT_IOU, device="cpu", verbose=False)
    preds = results[0]
    boxes = getattr(preds, "boxes", None)
    out = []
    if boxes is None or len(boxes) == 0:
        return out
    # robust extraction of arrays
    try:
        xyxy = boxes.xyxy.cpu().numpy()
    except Exception:
        xyxy = np.array(boxes.xyxy)
    try:
        confs = boxes.conf.cpu().numpy()
    except Exception:
        confs = np.array(boxes.conf)
    for bb, conf in zip(xyxy, confs):
        x1,y1,x2,y2 = map(int, bb[:4])
        w = x2 - x1; h = y2 - y1
        out.append((x1,y1,w,h,float(conf)))
    return out

def find_plate_candidates_by_contour(frame):
    """
    Contour-based plate proposals (fallback). Returns list (x,y,w,h).
    """
    h, w = frame.shape[:2]
    area_tot = h * w
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    try:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    except Exception:
        pass
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,3))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)
    edges = cv2.Canny(closed, 50, 200)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT,(5,3)), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand = []
    for c in cnts:
        x,y,ww,hh = cv2.boundingRect(c)
        if ww < 30 or hh < 12: continue
        rect_area = ww * hh
        if rect_area < area_tot * MIN_PLATE_AREA_FRAC: continue
        ar = ww / float(hh + 1e-9)
        if ar < PLATE_ASPECT_MIN or ar > PLATE_ASPECT_MAX: continue
        area = cv2.contourArea(c)
        solidity = area / float(rect_area + 1e-9)
        if solidity < 0.25: continue
        cand.append((x,y,ww,hh,ar,solidity))
    cand.sort(key=lambda t: t[2]*t[3], reverse=True)
    # remove nested
    filtered = []
    for (x,y,ww,hh,ar,sol) in cand:
        skip = False
        for (fx,fy,fw,fh,_,_) in filtered:
            if x >= fx and y >= fy and x+ww <= fx+fw and y+hh <= fy+fh:
                skip = True; break
        if not skip:
            filtered.append((x,y,ww,hh,ar,sol))
    return filtered

# ------------------ Warp perspective (rectify) ------------------
def rectify_plate(img, bbox):
    """
    Try to find best contour inside bbox and compute a perspective warp
    to produce a rectangular, horizontally aligned crop.
    If fails, return center-cropped, resized image.
    """
    x,y,w,h = bbox
    pad_x = int(w*0.2); pad_y = int(h*0.2)
    x0 = max(0, x - pad_x); y0 = max(0, y - pad_y)
    x1 = min(img.shape[1], x + w + pad_x); y1 = min(img.shape[0], y + h + pad_y)
    roi = img[y0:y1, x0:x1].copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # look for approx polygon with 4 corners and wide aspect
    best = None; best_area = 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            bx,by,bw,bh = cv2.boundingRect(approx)
            ar = bw / float(bh + 1e-9)
            if ar < 1.3: continue
            area = bw*bh
            if area > best_area:
                best_area = area; best = approx
    if best is not None:
        pts = best.reshape(4,2)
        # order points (tl, tr, br, bl)
        s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
        tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]; bl = pts[np.argmax(diff)]
        rect = np.array([tl, tr, br, bl], dtype="float32")
        # compute width/height
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxW = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxH = max(int(heightA), int(heightB))
        dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(roi, M, (maxW, maxH))
        return warp
    # fallback: center crop scaled
    H, W = roi.shape[:2]
    scale = 1.6 if max(H,W) < 300 else 1.0
    if scale != 1.0:
        roi = cv2.resize(roi, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_CUBIC)
    return roi

# ------------------ OCR ensemble + scoring ------------------
def ocr_and_score(plate_img, save_dir=None):
    """
    input: plate_img (BGR rectified)
    returns best_plate, best_score, debug_list (list of tuples)
    """
    debug = []
    best_plate = None; best_score = 0.0

    # create variants (gamma/clahe/threshold/sharpen)
    variants = []
    H,W = plate_img.shape[:2]
    scale = 1.0
    if max(H,W) < 300: scale = 1.8
    base = cv2.resize(plate_img, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_CUBIC) if scale != 1.0 else plate_img.copy()
    variants.append(base)
    # CLAHE
    try:
        lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB); l,a,b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)); l2 = clahe.apply(l)
        lab2 = cv2.merge((l2,a,b)); variants.append(cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR))
    except Exception:
        pass
    # gamma
    for g in (0.7, 1.2):
        inv = 1.0/g; table = np.array([((i/255.0)**inv)*255 for i in np.arange(256)]).astype("uint8")
        variants.append(cv2.LUT(base, table))
    # adaptive threshold
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    for b in (31, 41):
        try:
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, b, 9)
            variants.append(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))
        except Exception:
            pass
    # sharpen
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    try:
        variants.append(cv2.filter2D(base, -1, kernel))
    except Exception:
        pass

    v_idx = 0
    for v in variants:
        v_idx += 1
        # save variant if requested
        if save_dir is not None:
            try:
                fn = save_dir / f"var_{v_idx}.jpg"; cv2.imwrite(str(fn), v)
            except Exception:
                pass
        # run OCR backends
        hits = []
        hits += run_paddleocr(v)   # list of (txt, conf)
        hits += run_easyocr(v)
        hits += run_tesseract(v)
        for txt, conf in hits:
            txt_norm = normalize_plate_text(txt)
            regex_bonus = plate_regex_score(txt_norm)
            combined = float(conf) * 0.6 + regex_bonus * 0.4
            # penalize too-short
            if len(re.sub(r'[^A-Za-z0-9]', '', txt_norm)) < 4:
                combined *= 0.65
            debug.append((v_idx, txt, txt_norm, float(conf), regex_bonus, float(combined)))
            if combined > best_score:
                best_score = combined; best_plate = txt_norm

    return best_plate, float(best_score), debug

# ------------------ Everything together: process_frame ------------------
def process_frame(frame, yolo_plate_model=None, save_debug_dir=None):
    """
    Detect plate (by yolo or contour), rectify, OCR, return list of candidate (plate, score, bbox)
    """
    candidates = []
    # 1) plate detection by YOLO (if model provided)
    if yolo_plate_model is not None and YOLO is not None:
        yolo_boxes = detect_plates_with_yolo(frame, yolo_plate_model)
        for (x,y,w,h,conf) in yolo_boxes:
            # discard too small
            if w*h < frame.shape[0]*frame.shape[1]*MIN_PLATE_AREA_FRAC: continue
            candidates.append((x,y,w,h,conf))
    # 2) contour proposals (always run as fallback/extra)
    contours = find_plate_candidates_by_contour(frame)
    for (x,y,w,h,ar,sol) in contours:
        candidates.append((x,y,w,h, 0.0))

    # de-duplicate overlapping candidates by IoU/hierarchical
    boxes = []
    for (x,y,w,h,conf) in candidates:
        boxes.append({"x":x,"y":y,"w":w,"h":h,"conf":conf})
    # simple NMS-like: keep if not contained by larger
    boxes_sorted = sorted(boxes, key=lambda b: b["w"]*b["h"]*(1+b["conf"]), reverse=True)
    final = []
    for b in boxes_sorted:
        bx = (b["x"], b["y"], b["x"]+b["w"], b["y"]+b["h"])
        contained = False
        for f in final:
            fx = (f["x"], f["y"], f["x"]+f["w"], f["y"]+f["h"])
            if b["x"] >= f["x"] and b["y"] >= f["y"] and b["x"]+b["w"] <= f["x"]+f["w"] and b["y"]+b["h"] <= f["y"]+f["h"]:
                contained = True; break
        if not contained:
            final.append(b)

    outputs = []
    for idx, b in enumerate(final[:8]):  # limit top-k
        x,y,w,h, conf = b["x"],b["y"],b["w"],b["h"],b.get("conf",0.0)
        warp = rectify_plate(frame, (x,y,w,h))
        # debug dir for event
        debug_dir = None
        if save_debug_dir is not None:
            debug_dir = save_debug_dir / f"cand_{idx}_{uuid.uuid4().hex[:6]}"
            debug_dir.mkdir(parents=True, exist_ok=True)
        plate, score, debug_info = ocr_and_score(warp, save_dir=debug_dir)
        outputs.append({"plate":plate,"score":score,"bbox":(x,y,w,h),"warp":warp,"debug":debug_info, "det_conf":conf})
    return outputs

# ------------------ Persist event ------------------
def append_event_csv(ts_iso, plate, score, crop_path):
    header = not EVENTS_CSV.exists()
    with open(EVENTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header:
            w.writerow(["timestamp","plate","score","crop_path"])
        w.writerow([ts_iso, plate or "", f"{score:.3f}", str(crop_path)])

# ------------------ Main run loop ------------------
def run(video_src=VIDEO_SRC_DEFAULT, plate_yolo_weights=None, debug=CV_DEBUG):
    global count_entered, seen_plates
    # prepare model if weight provided
    yolo_plate_model = None
    if plate_yolo_weights and YOLO is not None:
        try:
            yolo_plate_model = YOLO(plate_yolo_weights)
            log("Loaded plate YOLO weights:", plate_yolo_weights)
        except Exception as e:
            log("Cannot load plate weights:", e)
            yolo_plate_model = None
    elif plate_yolo_weights:
        log("ultralytics not installed; cannot use plate YOLO model.")

    # init OCR backends
    ensure_paddle(); ensure_easyocr()

    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        log("Cannot open video source:", video_src); return

    frame_i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); time.sleep(0.2); continue
        frame_i += 1
        if frame_i % max(1, FRAME_SKIP) != 0: continue

        # prepare debug save dir per frame if needed
        save_dir = None
        if SAVE_VARIANTS:
            save_dir = OUTPUT_DIR / "debug_frame_" + str(int(time.time()))
            save_dir.mkdir(parents=True, exist_ok=True)

        outputs = process_frame(frame, yolo_plate_model=yolo_plate_model, save_debug_dir=save_dir)

        for out in outputs:
            plate = out["plate"]; score = out["score"]; warp = out["warp"]
            if plate and score >= OCR_SCORE_THRESHOLD:
                now = time.time()
                last = seen_plates.get(plate)
                if last is None or (now - last) > PLATE_SEEN_COOLDOWN:
                    count_entered += 1
                    seen_plates[plate] = now
                    # save warp crop
                    fname = OUTPUT_DIR / f"crop_{int(time.time())}_{plate}.jpg"
                    try: cv2.imwrite(str(fname), warp)
                    except: fname = OUTPUT_DIR / f"crop_{int(time.time())}_idx.jpg"; cv2.imwrite(str(fname), warp)
                    append_event_csv(time.strftime("%Y-%m-%dT%H:%M:%S"), plate, score, fname)
                    log(f"[NEW] plate={plate} score={score:.3f} total_entered={count_entered}")
                else:
                    # update last seen
                    seen_plates[plate] = now

        # debug overlay
        if debug:
            disp = frame.copy()
            for out in outputs:
                x,y,w,h = out["bbox"]
                plate = out["plate"] or ""
                score = out["score"]
                cv2.rectangle(disp, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(disp, f"{plate} {score:.2f}", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(disp, f"Entered: {count_entered}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.imshow("ANPR improved", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    if debug: cv2.destroyAllWindows()

# ------------------ CLI ------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video","-v", default=VIDEO_SRC_DEFAULT, help="video path or camera index")
    p.add_argument("--weights","-w", default=os.environ.get("PLATE_YOLO_WEIGHTS",""), help="plate YOLO weights (optional)")
    p.add_argument("--no-debug", action="store_true", help="turn off debug")
    args = p.parse_args()
    video = args.video
    try: video = int(video)
    except: pass
    debug = not args.no_debug
    run(video, plate_yolo_weights=args.weights or None, debug=debug)

if __name__ == "__main__":
    main()