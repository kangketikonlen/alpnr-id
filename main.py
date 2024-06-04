import os
import argparse
import cv2
from cv2 import dnn_superres
from ultralytics import YOLO
from reader import read_license_plate

IMAGES_PATH = './images'
COCO_MODEL_PATH = './models/yolov8n.pt'
DETECTOR_MODEL_PATH = './models/alpr-id.pt'
SR_MODEL_PATH = './models/edsr-x4.pb'

def setup_super_resolution(sr_model_path, scale=4):
    """Initialize and configure the super-resolution model."""
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(sr_model_path)
    sr.setModel("edsr", scale)
    return sr

def load_models():
    """Load YOLO models for object detection and license plate recognition."""
    coco_model = YOLO(COCO_MODEL_PATH)
    detector_model = YOLO(DETECTOR_MODEL_PATH)
    return coco_model, detector_model

def save_image(image, path):
    """Save the image to the specified path."""
    cv2.imwrite(path, image)

def process_image(sample_path, images_path, sr_model):
    """Process the sample image to detect and read the license plate."""
    # Read image
    img = cv2.imread(sample_path)
    
    # Detect license plates
    _, detector_model = load_models()
    license_plates = detector_model(img, verbose=False)[0]
    x1, y1, x2, y2, score, class_id = license_plates.boxes.data.tolist()[0]
    
    # Crop license plate
    license_plate_crop = img[int(y1):int(y2), int(x1):int(x2), :]
    save_image(license_plate_crop, f"{images_path}/license_plate.jpg")
    
    # Convert to grayscale
    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    save_image(license_plate_crop_gray, f"{images_path}/license_plate_gray.jpg")
    
    # Apply thresholding
    _, license_plate_crop_thresh = cv2.threshold(
        license_plate_crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    save_image(license_plate_crop_thresh, f"{images_path}/license_plate_thresh.jpg")
    
    # Read license plate
    try:
        read_license_plate(license_plate_crop_thresh)
    except Exception as e:
        print(f"Plat nomor tidak terbaca: {e}")
        return

def main(sample_path):
    # Check if the sample path exists
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Sample path '{sample_path}' does not exist.")
    
    # Setup super resolution model
    sr_model = setup_super_resolution(SR_MODEL_PATH)
    
    # Process the sample image
    process_image(sample_path, IMAGES_PATH, sr_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process an image to detect and read license plates.')
    parser.add_argument('-s', '--sample', required=True, help='Path to the sample image')
    args = parser.parse_args()
    
    main(args.sample)
