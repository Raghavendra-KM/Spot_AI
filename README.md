# SpotAI

## Overview
This project demonstrates an AI-powered object detection application using YOLOv5. The system takes an input image, identifies objects within it, and annotates the image with bounding boxes and labels for detected objects. The application is powered by PyTorch and OpenCV.

## Features
- **Real-Time Object Detection**: Detect objects in images using the YOLOv5 model.
- **High Accuracy**: Uses the `yolov5x` model for better object detection and classification.
- **Dynamic Text Labeling**: Ensures that object labels and bounding boxes fit within the image dimensions.
- **User-Friendly Input**: Accepts user-defined image paths for easy testing.
- **Customizable**: Supports modifications for detecting additional objects or improving accuracy.

## Prerequisites
Before running the project, ensure you have the following installed:

- Python 3.8 or higher
- pip (Python package manager)
- PyTorch
- OpenCV

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ai-object-detection.git
   cd ai-object-detection
   ```

2. Install required dependencies:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install opencv-python
   ```

3. Set up YOLOv5:
   The model will automatically download when you run the script for the first time using PyTorch's `torch.hub.load` function.

## Usage
1. Place your input image in the project directory or note its path.
2. Run the script:
   ```bash
   python object_detection.py
   ```
3. Enter the path to your input image when prompted.
4. The processed image with detected objects will be saved in the same directory as the input image, with "_detected" appended to its name.

## Example
### Input Image & Output Image
![Example](https://imgur.com/a/pphbkOo))

## Code Highlights
- **YOLOv5 Model**: The `yolov5x` model is loaded via PyTorch Hub for state-of-the-art object detection.
- **Dynamic Text Handling**: The code ensures text labels are displayed without exceeding image boundaries.
- **Resizing**: Large images are resized for optimal processing.

## Customization
- To use a different YOLOv5 model, replace `'yolov5x'` with `'yolov5s'`, `'yolov5m'`, or other available variants.
- Modify the confidence threshold for detection within the script to fine-tune results.

## Future Enhancements
- Add support for video input.
- Deploy the application as a web service using Flask or FastAPI.
- Extend support for training a custom YOLOv5 model for domain-specific tasks.

## License
This project is licensed under the MIT License.

## Acknowledgments
- [Ultralytics](https://github.com/ultralytics/yolov5) for providing the YOLOv5 framework.
- The open-source community for supporting PyTorch and OpenCV.
