import cv2
import numpy as np
import os

from ultralytics import YOLO
from module.sort import *
from util import get_car, read_license_plate, write_csv

SAMPLE_PATH = './samples'
SAMPLE_NAME = 'hikvision.mp4'

REPORT_PATH = './report'
REPORT_NAME = 'test.csv'
IMAGES_PATH = './images'  # Path to save images

# Create directory to save images if it doesn't exist
os.makedirs(IMAGES_PATH, exist_ok=True)

results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('./models/yolov8n.pt')
detector_model = YOLO('./models/alpr-id.pt')

# load video
cap = cv2.VideoCapture(os.path.join(SAMPLE_PATH, SAMPLE_NAME))

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    
    if ret:
        results[frame_nmr] = {}
        # Save the current frame for debugging
        cv2.imwrite(f"{IMAGES_PATH}/frame_{frame_nmr}.jpg", frame)
        print(f"Processing frame {frame_nmr}")

        # detect vehicles
        detections = coco_model(frame)[0]
        print(f"Detections: {detections}")

        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])
        
        print(f"Filtered detections: {detections_}")

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))
        print(f"Track IDs: {track_ids}")

        # detect license plates
        license_plates = detector_model(frame)[0]
        print(f"License plates: {license_plates}")

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            print(f"Assigned car ID: {car_id}")

            if car_id != -1:
                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                
                # Save the cropped license plate for debugging
                cv2.imwrite(f"{IMAGES_PATH}/frame_{frame_nmr}_license_plate_{car_id}.jpg", license_plate_crop)
                
                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Save the processed license plate for debugging
                cv2.imwrite(f"{IMAGES_PATH}/frame_{frame_nmr}_license_plate_thresh_{car_id}.jpg", license_plate_crop_gray)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray)
                print(f"License plate text: {license_plate_text}, Score: {license_plate_text_score}")

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {
                        'car': {
                            'bbox': [xcar1, ycar1, xcar2, ycar2]
                        },
                        'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': license_plate_text,
                        'bbox_score': score,
                        'text_score': license_plate_text_score
                        }
                    }

# write results
write_csv(results, os.path.join(REPORT_PATH, REPORT_NAME))
