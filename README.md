# Face Mask Detection with Real-Time Alert System

This project detects whether a person is wearing a **face mask** or **not** using a **deep learning model** and displays results through **live video feed** using **Flask + OpenCV**.  
The system alerts visually when **No Mask** is detected.

---

## 🎯 Objective
- Detect face mask usage in **real-time** using webcam.
- Classify faces into **Mask** or **No Mask**.
- Show live results in browser using Flask web interface.

---

## 🛠️ Tech Stack
| Component | Tool |
|----------|------|
| Programming Language | Python |
| Computer Vision | OpenCV |
| Model Training | TensorFlow / Keras |
| Web Framework | Flask |
| Dataset | Mask & No Mask Images (Kaggle) |

---

## 🎒 Dataset
Face Mask Detection Dataset
https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
## Alert System
If mask is detected → Green bounding box
If no mask is detected → Red bounding box & alert text shown
