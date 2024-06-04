# Automatic Indonesian License Plate Number Recognition

## Requirements
- Python 3.11.7
- OpenCV
- Ultralytics
- PaddleOCR

## Images Requirements
- Resolution 2560 x 1920 pixels
- Daylight
- The vehicle is +- 1 meter from the camera

## How to Run
Create virtual environment using python venv
```bash
python -m venv .venv
```
Enable your virtual environment.
```bash
source .venv/bin/activat
```
Install requirements using pip.
```bash
pip install -r requirements
```
Run main.py using python.
```bash
python main.py -s <image_path>
```

## Notes
- I use dataset from here [Vinlden Computer Vision Project](https://universe.roboflow.com/s1-upflo/viniden)
- You can see the training scripts in [google colab](https://colab.research.google.com/drive/1_Kp7KEURw08YN7XyyOw1P0eGZmQXcanw?usp=sharing)
- I use EDSR-x4 for better resolutions, unfortunately the number plate distorted.
- This project doesn't have error handling so use it careful.