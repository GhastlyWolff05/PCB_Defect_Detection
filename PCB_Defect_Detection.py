# 1. IMPORT ALL REQUIRED LIBRARIES, PACKAGES AND DEPENDENCIES

import cv2
import torch
import os
import time
from ultralytics import YOLO
from roboflow import Roboflow


rf = Roboflow(api_key)
project = rf.workspace("practice-w8vd6").project("simplepcb_defect_detection")
version = project.version(2)
dataset = version.download("yolov8")

# 2. TRAINING THE MODEL
# Load the YOLOv8s ARCHITECTURE
model = YOLO('yolov8s.yaml') 

# Train with Early Stopping
model.train(
    data=f"{dataset.location}/data.yaml", 
    epochs=150,
    patience=50,
    imgsz=640, 
    pretrained=False, 
    save=True,
    plots=True
)

# 3. REAL TIME VIDEO DETECTION
# 3.1. SETUP & CONFIGURATION
MODEL_PATH = '/content/runs/detect/train/weights/best.pt' 
INPUT_VIDEO = '/content/drive/MyDrive/PCB_Defect_Detection/Test_Videos/testVid_dedicated_test_v1.mp4'
OUTPUT_VIDEO = '/content/simplePCB_analysis_output.mp4'

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

model = YOLO(MODEL_PATH)

# 3.2. MODEL METRICS
if os.path.exists(MODEL_PATH):
    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"Model Size: {model_size_mb:.2f} MB")
else:
    print(f"CRITICAL: Model not found at {MODEL_PATH}")

try:
    metrics = model.val() 
    print(f"Model mAP@50: {metrics.box.map50:.4f}")
except:
    print("Validation metrics skipped.")

# 3.3. SEVERITY LOGIC
def assess_severity(class_name, conf):
    class_name = class_name.lower()
    if "missing" in class_name or "deep" in class_name:
        return "CRITICAL", (0, 0, 255) # Red
    if "scratched" in class_name or "discolored" in class_name:
        return "MINOR", (0, 255, 255) # Yellow
    if any(k in class_name for k in ['clk', 'mcu', 'usb']):
        return "PASS", (0, 255, 0) # Green
    return "DETECTED", (255, 255, 255)

# 3.4. VIDEO PROCESSING SETUP
cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    cap = cv2.VideoCapture(INPUT_VIDEO, cv2.CAP_FFMPEG)

fps_input = int(cap.get(cv2.CAP_PROP_FPS))
if fps_input == 0: fps_input = 30 

# INITIALIZE WRITER WITH TARGET DIMENSIONS
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps_input, (TARGET_WIDTH, TARGET_HEIGHT))

frame_count = 0
start_time = time.time()

print(f"\nStarting Video Inference... Output: {TARGET_WIDTH}x{TARGET_HEIGHT}")

# 3.5. PROCESSING LOOP
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
    frame_count += 1
    
    results = model(frame, verbose=False)[0]
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = model.names[cls]

        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        severity, color = assess_severity(class_name, conf)
        
        # Draw Bounding Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Label Text (Class | Conf | Severity)
        label_text = f"{class_name} {conf:.2f} | {severity}"
        t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - 22), (x1 + t_size[0], y1), color, -1)
        cv2.putText(frame, label_text, (x1, y1 - 7), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Show coordinates only for defects (Scratched, Missing, Discolored)
        if any(defect in class_name.lower() for defect in ["scratched", "missing", "discolored"]):
            # 1. Draw a high-visibility center dot (Red with white border)
            cv2.circle(frame, (center_x, center_y), 6, (255, 255, 255), -1) # Outer white
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)     # Inner red
            
            # 2. Draw the numerical coordinates (X, Y) next to the point
            coord_text = f"({center_x}, {center_y})"
            cv2.putText(frame, coord_text, (center_x + 8, center_y + 1), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, coord_text, (center_x + 7, center_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            print(f"Frame {frame_count} | {class_name} @ {coord_text} | {severity}")

    fps = frame_count / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    out.write(frame)

# 6. FINALIZATION
cap.release()
out.release()

print(f"\nProcessing Complete! File saved to: {OUTPUT_VIDEO}")

# 3.7. AUTOMATIC DOWNLOAD OF OUTPUT VIDEO
if os.path.exists(OUTPUT_VIDEO) and os.path.getsize(OUTPUT_VIDEO) > 0:
    from google.colab import files
    files.download(OUTPUT_VIDEO)
