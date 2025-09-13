from pathlib import Path
import cv2
from worker_yolo import recognize_plate_local_easyocr_improved

img = cv2.imread("/Users/hoangquannguyen/Downloads/haha.png")
plate, conf = recognize_plate_local_easyocr_improved(img)
print("Result:", plate, conf)