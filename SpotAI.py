import torch
import cv2
import numpy as np
from pathlib import Path

# Load a more robust YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)  # Use 'yolov5x' for better accuracy and more object classes

def detect_objects(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image. Please check the file path.")
        return
    
    # Resize image if it's too large to ensure labels fit
    max_dimension = 800
    height, width, _ = img.shape
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        img = cv2.resize(img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    
    # Perform inference
    results = model(img)  # Use the resized image directly
    
    # Process results
    results.print()  # Print results to the console
    
    # Draw bounding boxes and labels on the image
    labels, coords = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
    for label, coord in zip(labels, coords):
        x1, y1, x2, y2, conf = coord
        x1, y1, x2, y2 = int(x1 * img.shape[1]), int(y1 * img.shape[0]), int(x2 * img.shape[1]), int(y2 * img.shape[0])
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label and confidence
        label_text = f"{model.names[int(label)]} {conf:.2f}"
        
        # Ensure text fits within the image boundaries
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = max(x1, 0)
        text_y = max(y1 - 10, 0)
        if text_x + text_size[0] > img.shape[1]:
            text_x = img.shape[1] - text_size[0] - 5
        if text_y - text_size[1] < 0:
            text_y = text_size[1] + 5
        
        cv2.putText(img, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save the output image
    output_path = Path(image_path).stem + "_detected.jpg"
    cv2.imwrite(output_path, img)
    print(f"Output saved to {output_path}")

    # Show the output image (optional)
    cv2.imshow("Detected Objects", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Get image path from user
image_path = input("Enter the path of the image: ").strip()
detect_objects(image_path)
