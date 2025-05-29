# MWPD-Pothole-Detection
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

This project provides an AI-powered solution for detecting potholes on roads using images or videos. The core idea is to improve road safety and assist city authorities by identifying potholes automatically using computer vision.

Built using **YOLOv5**, the model has been trained on the **Multi-Weather Pothole Detection (MWPD)** dataset to recognize potholes under various lighting and weather conditions such as rain, fog, and shadows.

---

## 🌦️ Multi-Weather Pothole Detection (MWPD)

The **MWPD dataset** contains annotated images of potholes captured in real-world scenarios and different environmental conditions. It enhances the model's robustness and real-time detection capability in variable outdoor settings. This diversity in training data allows the model to perform well even when visibility is challenging due to weather.

---

## 🧠 YOLOv5

**YOLOv5 (You Only Look Once)** is a real-time object detection algorithm that detects objects with high accuracy in a single neural network pass. Its benefits include:

- Faster training and inference
- Lightweight and optimized architecture
- Customizable for different object classes (e.g., potholes)
- Strong community and open-source support

In this project, we fine-tuned a YOLOv5 model specifically to detect potholes from road imagery.

---

## 🏗️ Ultralytics Framework

This project utilizes the official [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) implementation. Ultralytics provides:

- A ready-to-use PyTorch-based training pipeline
- Pretrained models
- Evaluation and visualization tools
- Support for custom datasets

The model training and inference pipelines are built upon this framework.

---

## ⚙️ Project Setup

### 1. Clone the YOLOv5 Repository

'''bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt


2. Add Your Files
Place your custom notebook (MWPD.ipynb), dataset configuration (data.yaml), and weights (best.pt) inside the directory as needed.

🧪 Training
To train your model on the MWPD dataset:

bash
Copy
Edit
python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5s.pt --cache
This trains YOLOv5 using the small variant (yolov5s.pt) as the base and saves the trained weights in the runs/ directory.

🎯 Inference
To detect potholes from an image or video using the trained model:

bash
Copy
Edit
python detect.py --weights runs/train/exp/weights/best.pt --img 640 --source path_to_image_or_video
Replace path_to_image_or_video with a valid image file, video, or webcam stream.

📊 Results
Training Output: YOLOv5 generates training results such as precision, recall, and loss curves.

Detection Output: The model outputs bounding boxes with confidence scores around potholes in the input media.
