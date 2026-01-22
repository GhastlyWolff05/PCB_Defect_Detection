# PCB_Defect_Detection

Computer Vision Engineer Assignment from Sapien Robotics

A computer vision pipeline trained to detect electronic components and surface-level defects on Printed Circuit Boards (PCBs).

## Project Assets & Media
Due to GitHub file size limits, the raw dataset and the recordings are hosted on Google Drive.

- [Full Dataset on Roboflow](https://app.roboflow.com/practice-w8vd6/simplepcb_defect_detection/browse?queryText=&pageSize=100&startingIndex=0&browseQuery=true)
-   Check out all the raw images, annotated images, and the augmented images on my Roboflow workspace.
-   PLEASE NOTE:
-   Images of Defective objects *start with the name of the defect* in them (eg. 'scratched_mc_12', 'discolored_clk_4', 'missing_usb_10')
-   Images of non-defective objects do not have any defect in their name, and *start with 'ok'* (eg. 'ok_11', 'ok_63')
- [Inference Screen Recordings:](https://drive.google.com/drive/folders/1wnWmFWhAgrT2fVPUPoitFbssS8pTQtdz?usp=sharing) Video evidence of the model detecting defects in real-time.

## Performance Metrics
Evaluated on a Tesla T4 GPU using the YOLOv8s architecture.

| Metric | Value |
| :--- | :--- |
| **Model Size** | 22.5 MB |
| **Inference Speed** | 10.1 ms (~99 FPS) |
| **mAP@50** | 0.2406 |
| **mAP@50-95** | 0.1177 |

For more results documentation, click [here](https://drive.google.com/drive/folders/1TfvZXD4U5El_7sSSyKna4xwI5sWo0yR_?usp=sharing)

---

## HARDWARE REQUIREMENTS AND DEPENDENCIES

### HARDWARE
- Laptop with 8 GB RAM, 512GB SSD, Intel i5 11th Gen CPU, Intel Iris Xe Integrated Graphics
- Phone with macro lens (around 2MP)
- 4 x ST Link V2 USB Debugger
- Scissors
- White paper
- Whiteboard marker
### DEPENDENCIES
- No GPU needed if you have access to Google Colab
- LabelImg
- Roboflow
- Ultralytics YOLO v8
- PyTorch


---

## Installation & Setup

### 1. Clone the repository
```bash
git clone [https://github.com/GhastlyWolff05/PCB_Defect_Detection.git](https://github.com/GhastlyWolff05/PCB_Defect_Detection.git)
cd SimplePCB-Detection
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Script
```bash
python PCB_Defect_Detection.py
```

---

## Project Chain of Thought (Workflow)

- I used the ST Link V2 usb debugger as the manufactured item for this problem. I have 4 pieces of this usb, out of which only 2 properly work as intended. So using the 2 working pieces as the subject, I went ahead and made a custom dataset of this usb debugger for the first part of the assignment.
- With approximately 360 images taken by myself on my phone (Samsung M33 5G with a 2MP macro lens), I renamed them to an appropriate format ('ok_1', 'ok_42', 'scratched_mcu_7', discolored_usb_12') for this part of the assignment. Taking pictures of the debugger from
	- different angles: Up top, phone tilted towards me, tilted to my front, to the left, and to the right. Also, all 4 side profiles.
	- different lighting scenarios: under direct sunlight, near direct sunlight, inside a room with open windows during the day, direct light and diffused light from a tubelight, a few casted shadows, phone flashlight...
	- different backgrounds / bases: dark green table, dark grey surface, light grey surface, black wood table, light blue surface, yellow book, orange surface, dusty light brown, and the palm of my hand.

- To simulate the defects, I used the 2 defective st link debuggers, as any damage to them while simulating the defects wouldn't cause an issue later on.
	- For missing component, I covered the component with a colour contrasting piece of paper, and labeled its position as a missing component. Only critical severity.
	- For scratched component, I used a pair of scissors to simulate gentle and deep scratches on all three components. I then labeled the component and the damage according to the severity. Light scratches are of moderate severity, and deeper ones are of critical severity.
	- For discolored component, I used a black marker and applied it over any metal part of the component, and labeled the severity with the component. Discoloration is usually a minor defect, so it is classified as minor severity.

- I then labeled object images with bounding boxes on LabelImg and exported them to Roboflow, and split the data into **70-20-10** (Train/Val/Test)
- Then, I applied the following augmentation Applied Noise, Rotations, Brightness adjustments, and Blur to increase model robustness.
- Then, configured Google Colab with a **Tesla T4 GPU**, and trained for **200 epochs** using a private Roboflow API key.
- Then, I wrote a Python script to process video input, downscaling to 1280x720 to ensure stable FPS and memory management.
- Finally, I verified performance using loss curves and mean Average Precision (mAP).

---

## Severity Logic:
The model distinguishes between component identification and critical failures:
CRITICAL: Missing components or deep Scratches.
MINOR: Light/Surface scratches or discoloration.
PASS: Identified normal components.

## Architecture:

YOLOv8s: Chosen for its Anchor-Free detection and the C2f module, which provides superior gradient flow compared to YOLOv5.
LabelImg: Used for object class labeling.
Roboflow: Service that enables dataset augmentation.
Google Colab: To utilize the Tesla T4 GPU (Python 3 Google Compute Engine Backend) for training the model.

## Changes / Fixes to be made in the Future:

- Use a different labeling format (currently using PascalVOC, and when converting it to YOLO txt Format, it is messing up with the bounding boxes, which in turn, is causing the augmentation to augment poorly annotated images, thereby resulting in poor detection accuracy.)
- More images in dataset, and use YOLO v8m for at least a 1000 image dataset.
- Write a python script to take images upon keyboard input and automatically name them according to the category needed. 


By Rohanta Shaw 🕶️
