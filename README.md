# Dog-Cat-Object-Detection-Using-YOLO
A real-world Dog &amp; Cat Object Detection system built using YOLO, with custom image annotation in Label Studio, model training in Google Colab, and image/video detection using Python and OpenCV.
# 🐱🐶 YOLO Dog & Cat Object Detection

A real-world Computer Vision project that detects and localizes **cats and dogs** in images and videos using a custom-trained **YOLO object detection model**.

The project covers the complete workflow from **dataset preparation and annotation to model training, validation, image detection, and video detection**.

---

## 🚀 Project Overview

This project uses YOLO to detect cats and dogs and draw bounding boxes around the detected objects.

Unlike image classification, which only predicts the class of an image, object detection identifies:

* **What** the object is
* **Where** the object is located
* **How confident** the model is about the prediction

### Supported Classes

| Class ID | Class |
| -------- | ----- |
| 0        | Cat   |
| 1        | Dog   |

---

## 🛠️ Technologies Used

* **Python**
* **YOLO / Ultralytics**
* **Label Studio**
* **Google Colab**
* **Anaconda**
* **OpenCV**
* **Jupyter Notebook**
* **GitHub**

---

## 🔄 Project Workflow

```text
Image Collection
       ↓
Dataset Preparation
       ↓
Image Annotation
       ↓
Label Studio
       ↓
YOLO Dataset Export
       ↓
Train / Validation Split
       ↓
YOLO Model Training
       ↓
Model Validation
       ↓
Best Model (best.pt)
       ↓
Image Detection
       ↓
Video Detection
       ↓
Final Detection Output
```

---

## 📂 Project Structure

```text
YOLO-Dog-Cat-Object-Detection/
│
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   │
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   │
│   └── data.yaml
│
├── models/
│   └── best.pt
│
├── test_images/
│
├── test_videos/
│
├── outputs/
│
├── Train_YOLO_Models.ipynb
│
├── detect.py
│
└── README.md
```

---

## 🏷️ Dataset Annotation

The images were annotated using **Label Studio**.

For every image:

1. The image was imported into Label Studio.
2. A bounding box was drawn around each cat.
3. A bounding box was drawn around each dog.
4. The appropriate class was assigned.
5. The annotations were reviewed.
6. The dataset was exported in YOLO-compatible format.

Example:

```text
Cat → Class ID 0
Dog → Class ID 1
```

---

## 📄 YOLO Label Format

YOLO uses the following format:

```text
class_id x_center y_center width height
```

Example:

```text
0 0.512 0.463 0.350 0.620
1 0.710 0.550 0.280 0.500
```

The coordinates are normalized between **0 and 1**.

---

## ⚙️ Installation

Create a Python environment using Anaconda:

```bash
conda create -n yolo python=3.11 -y
```

Activate the environment:

```bash
conda activate yolo
```

Install the required packages:

```bash
pip install ultralytics opencv-python
```

---

## ☁️ Training Using Google Colab

The model can be trained using Google Colab with GPU acceleration.

Install Ultralytics:

```python
!pip install -U ultralytics
```

Import YOLO:

```python
from ultralytics import YOLO
```

Load the model:

```python
model = YOLO("yolo11n.pt")
```

Train the model:

```python
results = model.train(
    data="/content/dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    project="/content/runs",
    name="cat_dog_detector"
)
```

> Training parameters can be changed depending on the dataset size, GPU memory, and required performance.

---

## 📊 Model Validation

After training, the model can be evaluated using the validation dataset.

```python
metrics = model.val(
    data="/content/dataset/data.yaml",
    imgsz=640
)

print(metrics)
```

Important evaluation metrics include:

* Precision
* Recall
* mAP@50
* mAP@50-95

Visual inspection of predictions is also performed to identify missed detections and incorrect bounding boxes.

---

## 🖼️ Image Detection

After training, the best model weights can be used for prediction.

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="test.jpg",
    conf=0.25,
    save=True
)
```

The output image contains:

* Bounding boxes
* Class names
* Confidence scores

---

## 🎥 Video Detection

The trained model can also process a new video.

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="input_video.mp4",
    conf=0.25,
    save=True
)
```

YOLO processes the video frame by frame and generates an annotated output video.

The output is normally stored inside the prediction/run output directory created by YOLO.

---

## 💻 Local Detection Using Anaconda

After downloading `best.pt`, the model can be used locally on Windows.

Example:

```bash
yolo predict model=best.pt source="input_video.mp4" conf=0.25 save=True
```

For an image:

```bash
yolo predict model=best.pt so
```
