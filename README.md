# 🚁 Drone Detection System
### Real-Time Object Detection using YOLOv8 · Label Studio · Raspberry Pi

<p align="center">
  <img src="https://img.shields.io/badge/Model-YOLOv8-00BFFF?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Platform-Raspberry%20Pi-C51A4A?style=for-the-badge&logo=raspberry-pi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Annotation-Label%20Studio-FF6B6B?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Language-Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-In%20Progress-yellow?style=for-the-badge"/>
</p>

<p align="center">
  <b>An end-to-end pipeline for detecting drones in real-time using custom-trained YOLO models, built from data annotation to edge deployment.</b>
</p>

---

## 📌 Project Overview

This project implements a **drone detection system** capable of identifying drones in video streams and images using a custom-trained YOLOv8 model. The pipeline covers everything from dataset creation and annotation using **Label Studio**, to training, evaluation, and deployment on a **Raspberry Pi** edge device.

> **Current Status:** Successfully detecting drones via phone camera feed. Raspberry Pi deployment is partially functional — currently halted due to hardware/resource constraints.
---

## 🗺️ Pipeline Overview

```
Raw Images → Label Studio Annotation → YOLO YAML Config → YOLOv8 Training → Inference
                                                                                  ↓
                                                                   Phone (✅ Working)
                                                                   Raspberry Pi (⚠️ Partial)
```


---

## 🔧 Full Pipeline Walkthrough

### Step 1 — Data Collection 📸

Images of drones were collected from various sources:
- Custom photographs from different angles, distances, and lighting conditions
- Open-source drone datasets
- Screen captures from drone footage videos

**Note:** I used Labelstudio.io

---

### Step 2 — Annotation with Label Studio 🏷️

All images were annotated using **[Label Studio](https://labelstud.io/)**, an open-source data labeling tool.

#### Setup Label Studio

```bash
pip install label-studio
label-studio start
```

Then open your browser at `http://localhost:8080`.

#### Annotation Process

1. **Create a new project** → Set labeling template to **Object Detection with Bounding Boxes**
2. **Import images** → Upload your drone image dataset
3. **Add label class** → e.g., `drone`
4. **Draw bounding boxes** around every drone in each image
5. **Export annotations** → Choose **YOLO format** export

> Label Studio exports a `.zip` containing:
> - `labels/` folder with `.txt` files (one per image)
> - Each `.txt` file contains: `class_id x_center y_center width height` (all normalized 0–1)

#### Example label file (`image_001.txt`)
```
0 0.512 0.438 0.124 0.098
0 0.721 0.310 0.089 0.076
```
> `0` = drone class, followed by normalized bounding box coordinates

---

### Step 3 — YAML Configuration File 📄

The `drone.yaml` file tells YOLO where your data is and what classes to detect.

```yaml
# drone.yaml — Dataset Configuration

path: /path/to/drone-detection/data   # Root dataset directory
train: images/train                    # Training images (relative to path)
val: images/val                        # Validation images (relative to path)

# Number of classes
nc: 1

# Class names
names:
  0: drone
```

### Step 4 — Training with YOLOv8 🏋️

Training was done using **Ultralytics YOLOv8**, the latest iteration of the YOLO family.

#### Install dependencies

```bash
pip install ultralytics
```

#### Train the model

```bash
yolo detect train \
  data=data/drone.yaml \
  model=yolov8n.pt \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  name=drone_detector
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model`   | `yolov8n.pt` | Nano model — fast, suitable for edge devices |
| `epochs`  | `100` | Number of training cycles |
| `imgsz`   | `640` | Input image size |
| `batch`   | `16`  | Images per batch (reduce if OOM) |

#### Training Results

After training, results are saved to `runs/detect/drone_detector/`:
- `best.pt` — Best weights (use this for inference)
- `last.pt` — Weights from the final epoch

---

### Step 5 — Inference on Phone 📱 ✅

Real-time detection was tested using a phone camera feed streamed to the model. This **currently works** reliably.

```python
from ultralytics import YOLO

model = YOLO("models/best.pt")

# Run on webcam / phone IP stream
results = model.predict(
    source="http://<phone-ip>:8080/video",  # IP Webcam app stream
    show=True,
    conf=0.5,
    save=True
)
```

> **Recommended app:** [IP Webcam (Android)](https://play.google.com/store/apps/details?id=com.pas.webcam) — streams camera as an MJPEG HTTP feed.

---

### Step 6 — Raspberry Pi Deployment ⚠️ (Partial)

The goal was to run inference directly on a **Raspberry Pi** (Model 3B/4) for a fully standalone edge detection system.

#### Installation on Raspberry Pi

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python dependencies
pip install ultralytics --break-system-packages

# Or use a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install ultralytics opencv-python-headless
```

#### Run detection

```python
# raspberry_pi/deploy.py
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)  # Pi Camera or USB webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)
    annotated = results[0].plot()

    cv2.imshow("Drone Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 🚀 Quickstart

```bash
# Clone the repo
git clone https://github.com/<ardur-priyanshu>/yolo-drone-vision.git
cd drone-detection

# Install dependencies
pip install -r requirements.txt

# Run inference on an image
python src/detect.py --source path/to/image.jpg --weights models/best.pt

# Run on webcam
python src/detect.py --source 0 --weights models/best.pt
```

---

## 🔮 Roadmap

- [x] Collect and annotate drone dataset using Label Studio
- [x] Configure YOLO YAML and train YOLOv8 model
- [x] Validate real-time detection via phone camera
- [-] Optimize model for Raspberry Pi (NCNN / TFLite export)
- [ ] Achieve stable real-time inference on Raspberry Pi 4
- [ ] 
---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) | Object detection model |
| [Label Studio](https://labelstud.io/) | Data annotation |
| [OpenCV](https://opencv.org/) | Image/video processing |
| [Raspberry Pi](https://www.raspberrypi.com/) | Edge deployment target |
| Python 3.10+ | Core language |

---

## 📚 References

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [Label Studio Documentation](https://labelstud.io/guide/)
- [YOLOv8 on Raspberry Pi Guide](https://docs.ultralytics.com/guides/raspberry-pi/)
- [YOLO Label Format Specification](https://docs.ultralytics.com/datasets/detect/)

---
