import os
import cv2
import numpy as np
import imutils
import easyocr

from cv2 import dnn_superres
from ultralytics import YOLO
from matplotlib import pyplot as plt
from paddleocr import PaddleOCR

SAMPLE_PATH = './samples'
SAMPLE_NAME = 'mobil1.jpg'

IMAGES_PATH = './images'

# Create an SR object - only function that differs from c++ code
sr = dnn_superres.DnnSuperResImpl_create()

# load models
coco_model = YOLO('./models/yolov8n.pt')
detector_model = YOLO('./models/alpr-id.pt')
sr.readModel('./models/edsr-x4.pb')
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("edsr", 4)

# read image
img = cv2.imread(os.path.join(SAMPLE_PATH, SAMPLE_NAME))

# detect license plates
license_plates = detector_model(img)[0]
for license_plate in license_plates.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = license_plate
    
    # crop license plate
    license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]
    cv2.imwrite(f"{IMAGES_PATH}/license_plate.jpg", license_plate_crop)
    
    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{IMAGES_PATH}/license_plate_gray.jpg", license_plate_crop_gray)
    
    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(f"{IMAGES_PATH}/license_plate_thresh.jpg", license_plate_crop_thresh)
    
    results = ocr.ocr(license_plate_crop_thresh, cls=True)

    for idx in results:
        res = results[idx]
        
        for line in res:
            print(line)