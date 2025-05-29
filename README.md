# 🕳️ Pothole Detection Using YOLOv5

Detecting potholes in diverse weather conditions using a custom-trained YOLOv5 model on the MWPD dataset.

---

## 📑 Table of Contents

- [Introduction](#introduction)
- [Multi-Weather Pothole Detection (MWPD)](#multi-weather-pothole-detection-mwpd)
- [YOLOv5](#yolov5)
- [Ultralytics Framework](#ultralytics-framework)
- [Project Setup](#project-setup)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)

---

## 📖 Introduction

This project offers an AI-powered solution to detect potholes on roads using real-time image or video input. By leveraging deep learning, specifically the YOLOv5 object detection algorithm, this system aims to assist in smart city infrastructure and road maintenance.

---

## 🌦️ Multi-Weather Pothole Detection (MWPD)

The **MWPD dataset** (Multi-Weather Pothole Detection) contains labeled images of potholes captured in various conditions—rain, fog, low light, and shadows. This improves the model’s ability to detect potholes under real-world circumstances, increasing reliability and robustness.

---

## 🧠 YOLOv5

**YOLOv5** (You Only Look Once, version 5) is a high-speed, high-accuracy object detection model. It offers:

- Real-time detection
- Efficient performance on both CPU and GPU
- Strong community support
- Pretrained weights and easy customization

In this project, YOLOv5 is fine-tuned for binary object detection (potholes vs. background) using the MWPD dataset.

---

## 🏗️ Ultralytics Framework

This project is built using the [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) repository. Ultralytics provides:

- Easy-to-use APIs for training and inference
- Visualization tools
- Model evaluation metrics
- Support for custom datasets via `data.yaml`

---

## ⚙️ Project Setup

### 1. Clone the YOLOv5 Repository

git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt 

## 📁 Add Required Files

Before running training or inference, make sure the following files are correctly added:

- `MWPD.ipynb` – Jupyter notebook containing the full training and inference pipeline.
- `data.yaml` – Custom dataset configuration file specifying class names and dataset paths.
- `best.pt` – Trained YOLOv5 model weights obtained after training.
- `runs/` – (Optional) Contains training logs, weights, and result images from YOLOv5.

Organize your project structure as follows:
project-root/
│
├── yolov5/
│ ├── MWPD.ipynb
│ ├── data.yaml
│ ├── best.pt
│ └── ...


---

## 🧪 Training

Use the command below to train the YOLOv5 model on the MWPD dataset:

python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5s.pt --cache

Parameter Explanation:

--img 640 → Image size (can also use 416, 512)

--batch 16 → Batch size for training

--epochs 50 → Number of training epochs

--data data.yaml → Dataset configuration file

--weights yolov5s.pt → Pretrained base model

--cache → Caches images for faster training

🎯 Inference
To perform pothole detection using the trained model, run:

python detect.py --weights runs/train/exp/weights/best.pt --img 640 --source path_to_image_or_video
Examples of --source:

--source sample.jpg → Single image

--source test.mp4 → Video file

--source 0 → Webcam stream

The output will be saved in the runs/detect/exp/ folder by default.

📊 Results
The model predicts pothole regions with bounding boxes and confidence scores.

Works effectively in diverse weather and lighting conditions, as trained on MWPD.

Performance metrics (precision, recall, mAP) are logged in training output.


