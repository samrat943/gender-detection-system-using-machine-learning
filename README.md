# Gender Detection System using CNN

A complete **deep learning–based gender classification system** capable of detecting faces and predicting gender (Male/Female) in **real-time** using a webcam or on static images.  
This project uses a **Convolutional Neural Network (CNN)** and **OpenCV/cvlib** for face detection.

---

## 📌 Table of Contents
1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Project Structure](#project-structure)  
4. [Technologies Used](#technologies-used)  
5. [Installation](#installation)  
6. [Dataset Structure](#dataset-structure)  
7. [Model Training](#model-training)  
8. [Real-Time Detection](#real-time-detection)  
9. [Training Results](#training-results)

---

## 🧠 Project Overview
This project aims to build a machine learning system that can automatically detect faces and classify gender in real-time.  
The system:
- Trains a CNN model using labelled images  
- Detects faces from webcam feed  
- Predicts gender with probability  
- Displays bounding box + label on the face  

---

## ⭐ Features
✔️ End‑to‑end gender classification pipeline  
✔️ CNN model built using TensorFlow/Keras  
✔️ Real-time face detection using cvlib  
✔️ Training history plot (loss & accuracy)  
✔️ Webcam-based live prediction  
✔️ Easy-to-run modular scripts  

---

## 📁 Project Structure
```
📦 Gender Detection System
│
├── train.py                     # CNN model training script
├── detect_gender_webcam.py      # Real-time gender detection
├── gender_detection.model       # Saved trained model
├── plot.png                     # Training loss/accuracy graph
└── Gender_Detection_README.md   # Documentation
```

---

## 🛠 Technologies Used
- Python 3.8+
- TensorFlow / Keras  
- OpenCV  
- cvlib (for face detection)  
- NumPy  
- Matplotlib  
- Scikit‑learn  

---

## ⚙️ Installation

Install all dependencies:

```bash
pip install numpy opencv-python matplotlib tensorflow keras scikit-learn cvlib
```

If cvlib gives errors, try:

```bash
pip install cmake
pip install dlib
pip install cvlib
```

---

## 🗂 Dataset Structure  
Your dataset must follow this format:

```
gender_dataset_face/
│
├── man/
│   ├── image_01.jpg
│   ├── image_02.jpg
│   └── ...
│
└── woman/
    ├── image_01.jpg
    ├── image_02.jpg
    └── ...
```

Supported formats: `.jpg`, `.png`, `.jpeg`

---

## 🔧 Model Training

Run:

```bash
python train.py
```

### What `train.py` does:
- Loads all images and preprocesses them  
- Converts labels to binary (0 = woman, 1 = man)  
- Splits data into training & validation sets  
- Applies image augmentation  
- Builds & trains a CNN  
- Saves:
  - `gender_detection.model`
  - `plot.png` (accuracy/loss graph)

---

## 📊 Training Results  
The training script generates a graph:

- **Training Loss**  
- **Validation Loss**  
- **Training Accuracy**  
- **Validation Accuracy**

This plot helps visualize overfitting, stability, and overall model quality.

![Training Results Screenshot](https://i.imgur.com/TMLR4Zb.png)

---

## 🎥 Real-Time Detection

Run:

```bash
python detect_gender_webcam.py
```

### What happens:
- The webcam opens  
- cvlib detects face regions  
- Each face is cropped → resized to 96×96 → fed into model  
- Output labels shown on screen with confidence  
