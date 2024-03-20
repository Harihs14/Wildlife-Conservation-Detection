README.md:

# Wildlife Animal Detection using YOLOv8

This repository contains scripts for real-time wildlife animal detection using YOLOv8, a state-of-the-art object detection algorithm. By leveraging computer vision techniques, this project aims to contribute to wildlife conservation efforts by enabling the detection and monitoring of various animal species.

## Requirements
- Python 3.x
- OpenCV
- Ultralytics YOLO package
  
## Dataset
- https://universe.roboflow.com/machine-train-ur3hn/animals-detection-bsbbi.

## Installation
1. Clone this repository to your local machine.
2. Install the required dependencies using pip:
    ```
    pip install -r requirements.txt
    ```

## Usage
1. Run the Python script `animal.py` to start real-time detection from your webcam.
    ```
    python animal.py
    ```
2. Press the 'Esc' key to exit the detection loop.

## Customization
- You can modify the list of `classnames` in the script to include or exclude specific animal classes according to your needs.
- Replace the provided `best.pt` model with your own trained YOLOv8 model for customized detection.

## Acknowledgments
- The YOLOv8 model implementation is based on Ultralytics YOLO package.
- Sample code adapted from [Ultralytics YOLO documentation](https://github.com/ultralytics/yolov8).
