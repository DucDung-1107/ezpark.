"""
python anpr_plate_only.py --video /Users/hoangquannguyen/Downloads/xemay_sang.mov --debug --frame-skip 2 --ocr-threshold 0.35
"""

import os, sys, time, argparse, csv, re
from pathlib import Path
from collections import deque

import cv2
import numpy as np

try:
    import easyocr
except Exception:
    easyocr = None

try:
    from concurrent.futures import ThreadPoolExecutor
except Exception:
    ThreadPoolExecutor = None

try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except Exception:
    GPU_AVAILABLE = False

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./plate_evidence"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_CSV = OUTPUT_DIR / "events.csv"

FRAME_SKIP = int(os.environ.get("FRAME_SKIP", "2"))
DEBUG = False

MIN_AREA_FRAC = float(os.environ.get("MIN_AREA_FRAC", "0.0008"))  
ASPECT_MIN = float(os.environ.get("ASPECT_MIN", "2.0"))   
ASPECT_MAX = float(os.environ.get("ASPECT_MAX", "6.5"))   
SOLIDITY_MIN = float(os.environ.get("SOLIDITY_MIN", "0.3"))

OCR_SCORE_THRESHOLD = float(os.environ.get("OCR_SCORE_THRESHOLD", "0.35"))
PLATE_COOLDOWN = float(os.environ.get("PLATE_COOLDOWN", "30.0"))  
MAX_CANDIDATES = int(os.environ.get("MAX_CANDIDATES", "8"))
RESIZE_WIDTH = int(os.environ.get("RESIZE_WIDTH", "1280"))  # Resize frame for speed
ENABLE_GPU = os.environ.get("ENABLE_GPU", "auto").lower()  # auto, true, false  


# Updated regex patterns for Vietnamese license plates
_VN_PLATE_REGEX = re.compile(r'([0-9]{2,3})\s*[-]?\s*([A-Z0-9]{1,4})\s*[-]?\s*([0-9]{3,5})', re.I)
_VN_MOTORBIKE_REGEX = re.compile(r'([0-9]{2})\s*[-]?\s*([A-Z0-9]{1,2})\s*[-]?\s*([0-9]{3,5})', re.I)
_VN_CAR_REGEX = re.compile(r'([0-9]{2,3})\s*[-]?\s*([A-Z]{1,2})\s*[-]?\s*([0-9]{3,5})', re.I)


_seen_plates = {}   
_entered_count = 0


_easyocr_reader = None

# ---------------- Helpers ----------------
def log(*args):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), *args)

def normalize_plate_text(s):
    if not s: return s
    s2 = re.sub(r'[^A-Za-z0-9\.\-]', '', s).upper()
    s2 = re.sub(r'\.+', '.', s2)
    return s2

def plate_regex_score(text):
    """Enhanced regex scoring for Vietnamese license plates"""
    if not text: return 0.0
    txt = normalize_plate_text(text)
    
    # Check for complete VN plate patterns
    if _VN_PLATE_REGEX.search(txt):
        return 1.0
    if _VN_MOTORBIKE_REGEX.search(txt):
        return 0.95
    if _VN_CAR_REGEX.search(txt):
        return 0.95
    
    # Partial scoring based on character types and length
    alnum = len(re.sub(r'[^A-Za-z0-9]', '', txt))
    has_numbers = bool(re.search(r'[0-9]', txt))
    has_letters = bool(re.search(r'[A-Za-z]', txt))
    
    score = 0.0
    if alnum >= 5 and has_numbers and has_letters:
        score = min(0.7, 0.1 * alnum)
    elif alnum >= 4 and has_numbers:
        score = min(0.5, 0.08 * alnum)
    else:
        score = min(0.3, 0.05 * alnum)
    
    return score

def ensure_easyocr():
    """Initialize EasyOCR with Vietnamese support and GPU optimization"""
    global _easyocr_reader
    if easyocr is None:
        log("Error: easyocr chưa được cài. pip install easyocr và torch (CPU/GPU phù hợp).")
        return False
    if _easyocr_reader is None:
        # Determine GPU usage
        use_gpu = False
        if ENABLE_GPU == "true":
            use_gpu = True
        elif ENABLE_GPU == "auto":
            use_gpu = GPU_AVAILABLE
            
        log(f"Initializing EasyOCR with Vietnamese support. GPU: {use_gpu}")
        try:
            # Add Vietnamese language support for better accuracy
            _easyocr_reader = easyocr.Reader(['en', 'vi'], gpu=use_gpu)
        except Exception as e:
            log(f"Failed to init EasyOCR with Vietnamese, falling back to English only: {e}")
            _easyocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
    return True

# ---------------- Plate detection (contour based) ----------------
def find_plate_candidates(frame):
    """
    Enhanced plate detection with improved preprocessing pipeline.
    Returns list of bbox (x,y,w,h) for potential license plate regions.
    """
    h, w = frame.shape[:2]
    area_total = h * w

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Advanced denoising: bilateral filter preserves edges while reducing noise
    gray = cv2.bilateralFilter(gray, 11, 80, 80)
    
    # Adaptive histogram equalization (CLAHE) for better contrast
    try:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    except Exception:
        pass

    # Multiple morphological operations to better connect characters
    # Horizontal kernel to connect characters horizontally
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (13,3))
    morph_h = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_h, iterations=1)
    
    # Vertical kernel to clean up vertical noise
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3,7))
    morph_v = cv2.morphologyEx(morph_h, cv2.MORPH_OPEN, kernel_v, iterations=1)

    # Enhanced edge detection with multiple approaches
    # Canny with automatic threshold
    sigma = 0.33
    median = np.median(morph_v)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    edges = cv2.Canny(morph_v, lower, upper)
    
    # Dilate edges to better connect
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,3))
    edges = cv2.dilate(edges, dilate_kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    
    for c in contours:
        x,y,ww,hh = cv2.boundingRect(c)
        
        # Basic size filtering
        if ww < 30 or hh < 15:  # Increased minimum size
            continue
            
        rect_area = ww * hh
        if rect_area < area_total * MIN_AREA_FRAC:
            continue
            
        # Aspect ratio filtering
        ar = ww / float(hh + 1e-9)
        if ar < ASPECT_MIN or ar > ASPECT_MAX:
            continue
            
        # Solidity check (how filled the contour is)
        area = cv2.contourArea(c)
        solidity = area / float(rect_area + 1e-9)
        if solidity < SOLIDITY_MIN:
            continue
            
        # Extent check (ratio of contour area to bounding rectangle area)
        extent = area / float(rect_area + 1e-9)
        if extent < 0.2:  # Skip too sparse regions
            continue
            
        # Accept candidate
        candidates.append((x,y,ww,hh, ar, solidity, extent))
    
    # Sort by combined score: area * solidity * extent
    candidates.sort(key=lambda t: (t[2]*t[3]) * t[5] * t[6], reverse=True)
    
    # Advanced overlap removal using IoU
    filtered = []
    for (x,y,ww,hh,ar,sol,ext) in candidates:
        overlap = False
        for (fx,fy,fw,fh,_,_,_) in filtered:
            # Calculate Intersection over Union (IoU)
            x1, y1, x2, y2 = x, y, x+ww, y+hh
            fx1, fy1, fx2, fy2 = fx, fy, fx+fw, fy+fh
            
            ix1, iy1 = max(x1, fx1), max(y1, fy1)
            ix2, iy2 = min(x2, fx2), min(y2, fy2)
            
            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                union = (x2-x1)*(y2-y1) + (fx2-fx1)*(fy2-fy1) - intersection
                iou = intersection / float(union + 1e-9)
                
                if iou > 0.3:  # Significant overlap
                    overlap = True
                    break
                    
        if not overlap:
            filtered.append((x,y,ww,hh,ar,sol,ext))
    
    return filtered[:MAX_CANDIDATES]  # Limit number of candidates

# ---------------- OCR wrapper with preprocessing variants ----------------
def generate_variants(region):
    """Enhanced preprocessing variants for better OCR accuracy"""
    imgs = []
    H, W = region.shape[:2]
    
    # Adaptive scaling based on region size
    if max(H, W) < 200:
        scale = 2.0
    elif max(H, W) < 300:
        scale = 1.6
    else:
        scale = 1.2
        
    try:
        img0 = cv2.resize(region, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_CUBIC)
    except Exception:
        img0 = region.copy()
    imgs.append(img0)
    
    # Advanced denoising
    try:
        denoised = cv2.fastNlMeansDenoisingColored(img0, None, 10, 10, 7, 21)
        imgs.append(denoised)
    except Exception:
        pass
    
    # CLAHE on L channel (LAB color space)
    try:
        lab = cv2.cvtColor(img0, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        imgs.append(cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR))
    except Exception:
        pass
    
    # Gamma correction variants
    for gamma in (0.5, 0.7, 1.0, 1.3, 1.6):
        try:
            inv = 1.0/gamma
            table = np.array([((i/255.0)**inv)*255 for i in np.arange(256)]).astype("uint8")
            imgs.append(cv2.LUT(img0, table))
        except Exception:
            pass
    
    # Advanced thresholding
    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    
    # Adaptive threshold variants
    for block_size in (21, 31, 41):
        for c_value in (5, 9, 15):
            try:
                th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, block_size, c_value)
                imgs.append(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))
            except Exception:
                pass
    
    # Otsu thresholding
    try:
        _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        imgs.append(cv2.cvtColor(th_otsu, cv2.COLOR_GRAY2BGR))
    except Exception:
        pass
    
    # Morphological operations
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        imgs.append(cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR))
    except Exception:
        pass
    
    # Sharpening filters
    kernels = [
        np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),  # Standard sharpen
        np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]),  # Strong sharpen
        np.array([[0,-1,0],[-1,6,-1],[0,-1,0]])  # Medium sharpen
    ]
    
    for kernel in kernels:
        try:
            sharpened = cv2.filter2D(img0, -1, kernel)
            imgs.append(sharpened)
        except Exception:
            pass
    
    # Edge enhancement
    try:
        edges = cv2.Canny(gray, 50, 150)
        # Dilate edges slightly
        kernel = np.ones((2,2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        # Combine with original
        enhanced = cv2.addWeighted(img0, 0.8, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.2, 0)
        imgs.append(enhanced)
    except Exception:
        pass
    
    return imgs

def ocr_candidates(region, oid=None, save_dir=None):
    """
    Enhanced OCR with parallel processing and advanced scoring.
    region: BGR image containing plate region
    Return best_text (normalized), best_score (0..1), and debug_info list
    """
    global _easyocr_reader
    if easyocr is None:
        return None, 0.0, []

    if _easyocr_reader is None:
        if not ensure_easyocr():
            return None, 0.0, []

    best_text = None
    best_score = 0.0
    debug_list = []

    variants = generate_variants(region)
    
    def process_variant(var_data):
        var_idx, v = var_data
        results = []
        
        # Optional save
        if save_dir is not None:
            try:
                fn = save_dir / f"oid{oid}_var{var_idx}.jpg"
                cv2.imwrite(str(fn), v)
            except Exception:
                pass
        
        # Run EasyOCR
        try:
            res = _easyocr_reader.readtext(v, 
                                         width_ths=0.4,  # Minimum width threshold
                                         height_ths=0.4,  # Minimum height threshold
                                         paragraph=False,  # Don't group into paragraphs
                                         detail=1)  # Return detailed results
        except Exception as e:
            res = []
        
        for (bbox, txt, conf) in res:
            norm = normalize_plate_text(txt)
            if len(norm) < 3:  # Skip very short text
                continue
                
            regex_bonus = plate_regex_score(norm)
            
            # Enhanced scoring algorithm
            base_score = float(conf)
            
            # Bonus for regex pattern match
            pattern_bonus = regex_bonus * 0.5
            
            # Length bonus (prefer reasonable lengths)
            length = len(re.sub(r'[^A-Za-z0-9]', '', norm))
            if 5 <= length <= 12:
                length_bonus = 0.1
            elif 3 <= length <= 4:
                length_bonus = 0.05
            else:
                length_bonus = -0.1
            
            # Character type diversity bonus
            has_nums = bool(re.search(r'[0-9]', norm))
            has_letters = bool(re.search(r'[A-Za-z]', norm))
            diversity_bonus = 0.1 if (has_nums and has_letters) else 0.0
            
            # Confidence bonus for very high confidence
            conf_bonus = 0.1 if conf > 0.8 else 0.0
            
            # Final score
            score = base_score + pattern_bonus + length_bonus + diversity_bonus + conf_bonus
            score = max(0.0, min(1.0, score))
            
            results.append((var_idx, txt, norm, float(conf), regex_bonus, float(score)))
        
        return results
    
    # Process variants (with optional parallel processing)
    if ThreadPoolExecutor and len(variants) > 3:
        try:
            with ThreadPoolExecutor(max_workers=min(4, len(variants))) as executor:
                all_results = list(executor.map(process_variant, enumerate(variants, 1)))
                # Flatten results
                for result_list in all_results:
                    debug_list.extend(result_list)
        except Exception:
            # Fallback to sequential processing
            for var_idx, v in enumerate(variants, 1):
                debug_list.extend(process_variant((var_idx, v)))
    else:
        # Sequential processing
        for var_idx, v in enumerate(variants, 1):
            debug_list.extend(process_variant((var_idx, v)))
    
    # Find best result
    for (var_idx, txt, norm, conf, regex_bonus, score) in debug_list:
        if score > best_score:
            best_score = score
            best_text = norm
    
    # Clamp score
    best_score = max(0.0, min(1.0, float(best_score or 0.0)))
    return best_text, best_score, debug_list

# ---------------- Persist event ----------------
def append_event(ts_iso, plate, score, crop_path):
    header_needed = not EVENTS_CSV.exists()
    with open(EVENTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(["timestamp","plate","score","crop_path"])
        w.writerow([ts_iso, plate or "", f"{score:.3f}", str(crop_path)])

def run(video_src, debug=False, frame_skip=FRAME_SKIP, resize_width=RESIZE_WIDTH):
    """Enhanced main processing loop with performance optimizations"""
    global _entered_count, _seen_plates
    if easyocr is None:
        log("Warning: easyocr chưa cài. Cài bằng pip install easyocr và torch (CPU/GPU). OCR sẽ không chạy.")
    
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        log("Không mở được video/camera:", video_src)
        return

    # Get video properties for optimization
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate resize factor if needed
    resize_factor = 1.0
    if resize_width > 0 and original_width > resize_width:
        resize_factor = resize_width / original_width
        new_height = int(original_height * resize_factor)
        log(f"Resizing frames from {original_width}x{original_height} to {resize_width}x{new_height}")
    
    log(f"Video: {original_width}x{original_height} @ {fps:.1f}fps, resize_factor: {resize_factor:.2f}")

    frame_i = 0
    processing_time_buffer = deque(maxlen=30)  # Track processing times
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            # Loop video for testing
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.2)
            continue
            
        frame_i += 1
        if frame_i % max(1, frame_skip) != 0:
            continue

        # Resize frame for performance
        if resize_factor < 1.0:
            frame = cv2.resize(frame, (resize_width, int(original_height * resize_factor)))

        # Find candidates
        candidates = find_plate_candidates(frame)
        found_this_frame = []
        
        # Process top candidates
        for idx, candidate_data in enumerate(candidates[:MAX_CANDIDATES]):
            if len(candidate_data) == 7:  # New format with extent
                x, y, w, h, ar, sol, ext = candidate_data
            else:  # Old format compatibility
                x, y, w, h, ar, sol = candidate_data
                ext = 0.5
            
            # Enlarge ROI slightly to include full plate
            padding_x = int(w * 0.15)
            padding_y = int(h * 0.15)
            x0 = max(0, x - padding_x)
            y0 = max(0, y - padding_y)
            x1 = min(frame.shape[1], x + w + padding_x)
            y1 = min(frame.shape[0], y + h + padding_y)
            roi = frame[y0:y1, x0:x1]
            
            if roi.size == 0:
                continue
            
            # Prepare debug directory
            ts = int(time.time())
            evt_dir = OUTPUT_DIR / f"evt_{ts}_{idx}"
            if debug:
                evt_dir.mkdir(parents=True, exist_ok=True)
            
            # OCR processing
            plate, score, dbg = ocr_candidates(roi, oid=idx, save_dir=evt_dir if debug else None)
            
            # Save debug info
            if debug:
                with open(evt_dir / "debug.txt", "w", encoding="utf-8") as fh:
                    fh.write(f"frame={frame_i} candidate_idx={idx} bbox=({x},{y},{w},{h})\n")
                    fh.write(f"aspect_ratio={ar:.3f} solidity={sol:.3f} extent={ext:.3f}\n")
                    fh.write(f"plate={plate} score={score:.3f}\n")
                    fh.write("OCR Debug Results:\n")
                    for row in dbg:
                        fh.write(f"  {row}\n")
            
            # Check if valid plate detected
            if plate and score >= OCR_SCORE_THRESHOLD:
                now = time.time()
                last_seen = _seen_plates.get(plate)
                
                if last_seen is None or (now - last_seen) > PLATE_COOLDOWN:
                    _seen_plates[plate] = now
                    _entered_count += 1
                    
                    # Save crop
                    crop_fn = OUTPUT_DIR / f"crop_{ts}_{plate}_{score:.3f}.jpg"
                    try:
                        cv2.imwrite(str(crop_fn), roi)
                    except Exception:
                        crop_fn = OUTPUT_DIR / f"crop_{ts}_idx{idx}.jpg"
                        cv2.imwrite(str(crop_fn), roi)
                    
                    # Log event
                    append_event(time.strftime("%Y-%m-%dT%H:%M:%S"), plate, score, crop_fn)
                    log(f"[DETECT] Plate: {plate} | Score: {score:.3f} | Total: {_entered_count}")
                else:
                    # Update last seen
                    _seen_plates[plate] = now
                    
                found_this_frame.append((plate, score, (x, y, w, h)))
            else:
                # Save low-score candidates in debug mode
                if debug and score > 0:
                    try:
                        cv2.imwrite(str(evt_dir / f"candidate_{idx}_lowscore_{score:.3f}.jpg"), roi)
                    except Exception:
                        pass

        # Performance tracking
        processing_time = time.time() - start_time
        processing_time_buffer.append(processing_time)
        avg_processing_time = sum(processing_time_buffer) / len(processing_time_buffer)

        # Debug visualization
        if debug:
            disp = frame.copy()
            
            # Draw detected plates
            for (plate, score, (x, y, w, h)) in found_this_frame:
                cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(disp, f"{plate} ({score:.2f})", (x, y-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw all candidates
            for candidate_data in candidates[:MAX_CANDIDATES]:
                if len(candidate_data) >= 6:
                    x, y, w, h = candidate_data[:4]
                    cv2.rectangle(disp, (x, y), (x+w, y+h), (255, 0, 0), 1)
            
            # Performance info
            cv2.putText(disp, f"Entered: {_entered_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(disp, f"Frame: {frame_i}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(disp, f"Process: {avg_processing_time:.3f}s", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(disp, f"Candidates: {len(candidates)}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Enhanced Plate Detection", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if debug:
        cv2.destroyAllWindows()
    
    log(f"Processing completed. Total plates detected: {_entered_count}")
    log(f"Average processing time per frame: {avg_processing_time:.3f}s")

# ---------------- CLI entry ----------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Enhanced Vietnamese License Plate Recognition System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--video", "-v", default=0, 
                   help="Video file path or camera index")
    p.add_argument("--debug", action="store_true", 
                   help="Enable debug mode with visualization and detailed logging")
    p.add_argument("--frame-skip", type=int, default=FRAME_SKIP,
                   help="Process every Nth frame (higher = faster but less accurate)")
    p.add_argument("--resize-width", type=int, default=RESIZE_WIDTH,
                   help="Resize frame width for processing (0 = no resize)")
    p.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES,
                   help="Maximum candidates to process per frame")
    p.add_argument("--min-area-frac", type=float, default=MIN_AREA_FRAC,
                   help="Minimum area fraction for plate candidates")
    p.add_argument("--aspect-min", type=float, default=ASPECT_MIN,
                   help="Minimum aspect ratio for plate candidates")
    p.add_argument("--aspect-max", type=float, default=ASPECT_MAX,
                   help="Maximum aspect ratio for plate candidates")
    p.add_argument("--solidity-min", type=float, default=SOLIDITY_MIN,
                   help="Minimum solidity for plate candidates")
    p.add_argument("--ocr-threshold", type=float, default=OCR_SCORE_THRESHOLD,
                   help="Minimum OCR score threshold for plate detection")
    p.add_argument("--cooldown", type=float, default=PLATE_COOLDOWN,
                   help="Cooldown period (seconds) before re-detecting same plate")
    p.add_argument("--enable-gpu", choices=['auto', 'true', 'false'], 
                   default=ENABLE_GPU, help="GPU usage for EasyOCR")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR),
                   help="Output directory for evidence and logs")
    args = p.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    # Update globals from args
    DEBUG = args.debug
    FRAME_SKIP = max(1, args.frame_skip)
    RESIZE_WIDTH = args.resize_width
    MAX_CANDIDATES = max(1, args.max_candidates)
    MIN_AREA_FRAC = args.min_area_frac
    ASPECT_MIN = args.aspect_min
    ASPECT_MAX = args.aspect_max
    SOLIDITY_MIN = args.solidity_min
    OCR_SCORE_THRESHOLD = args.ocr_threshold
    PLATE_COOLDOWN = args.cooldown
    ENABLE_GPU = args.enable_gpu.lower()
    
    # Update output directory
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EVENTS_CSV = OUTPUT_DIR / "events.csv"

    # Convert video arg if numeric
    video_src = args.video
    try:
        video_src = int(video_src)
    except Exception:
        video_src = args.video

    # Initialize CSV
    if not EVENTS_CSV.exists():
        with open(EVENTS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "plate", "score", "crop_path"])

    # Check OCR library
    if not ensure_easyocr():
        log("EasyOCR không sẵn sàng — script sẽ chạy nhưng không đọc được biển.")
    
    # Log configuration
    log("=== Enhanced Vietnamese License Plate Recognition ===")
    log(f"Video source: {video_src}")
    log(f"Debug mode: {DEBUG}")
    log(f"Frame skip: {FRAME_SKIP}")
    log(f"Resize width: {RESIZE_WIDTH}")
    log(f"Max candidates: {MAX_CANDIDATES}")
    log(f"OCR threshold: {OCR_SCORE_THRESHOLD}")
    log(f"Cooldown: {PLATE_COOLDOWN}s")
    log(f"GPU enabled: {ENABLE_GPU}")
    log(f"Output directory: {OUTPUT_DIR}")
    log(f"Threading available: {ThreadPoolExecutor is not None}")
    
    # Start processing
    run(video_src, debug=DEBUG, frame_skip=FRAME_SKIP, resize_width=RESIZE_WIDTH)