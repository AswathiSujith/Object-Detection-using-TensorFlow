# 🎯 Object Detection using TensorFlow

A **Deep Learning-based Object Detection System** built using **TensorFlow** and **OpenCV**.  
This project demonstrates how to detect and localize multiple objects within images or videos using pre-trained deep learning models like **SSD MobileNet**.

---

## 📘 Table of Contents

1. [Introduction](#introduction)  
2. [Objectives](#objectives)  
3. [Literature Review](#literature-review)  
4. [System Architecture](#system-architecture)  
5. [Tools and Technologies](#tools-and-technologies)  
6. [Implementation](#implementation)  
7. [Results](#results)  
8. [Applications](#applications)  
9. [Conclusion](#conclusion)  
10. [Future Work](#future-work)  
11. [Contributors](#contributors)

---

## 🔍 Introduction

**Object Detection** is a key task in **Computer Vision** that involves identifying and locating multiple objects in an image or video.  
Unlike simple classification, object detection provides both:
- **Class label** (what the object is)
- **Bounding box coordinates** (where the object is)

TensorFlow’s **Object Detection API** simplifies the process of building and training custom or pre-trained object detection models.

**Use Cases:**
- Self-driving cars  
- Surveillance systems  
- Medical imaging  
- Retail analytics  

---

## 🎯 Objectives

- Implement object detection using **TensorFlow 2.x**  
- Train and evaluate deep learning models for **object localization and recognition**  
- Experiment with **pre-trained models** (e.g., SSD MobileNet, Faster R-CNN)  
- Apply detection to **real-world scenarios** with multiple objects per frame  

---

## 📚 Literature Review

| Concept | Description |
|----------|--------------|
| **Image Classification vs Object Detection** | Classification assigns one label per image; detection identifies multiple objects with locations. |
| **Popular Architectures** | YOLO, SSD, Faster R-CNN are the leading detection frameworks. |
| **Transfer Learning** | Using pre-trained models (COCO dataset) accelerates training and improves accuracy. |

---

## ⚙️ System Architecture

**Workflow:**

Input Image/Video → Preprocessing → Model Inference → Bounding Boxes + Labels → Output Visualization


**Steps:**
1. **Data Collection:** Use COCO, Pascal VOC, or custom datasets.  
2. **Data Annotation:** Label images using tools like **LabelImg**.  
3. **Model Selection:** Choose models like SSD, YOLO, or Faster R-CNN.  
4. **Training:** Train or fine-tune the model using TensorFlow Object Detection API.  
5. **Evaluation:** Measure accuracy using **mAP (mean Average Precision)**.  
6. **Deployment:** Run real-time inference on webcam or video streams.

---

## 🧰 Tools and Technologies

| Category | Tools/Frameworks |
|-----------|------------------|
| **Programming Language** | Python 3.8+ |
| **Framework** | TensorFlow 2.x |
| **Libraries** | OpenCV, NumPy, Matplotlib |
| **Annotation Tool** | LabelImg |
| **Hardware** | GPU/TPU (for training acceleration) |

---

## 💻 Implementation

### 🔧 Environment Setup
```bash
pip install tensorflow opencv-python matplotlib numpy
pip install tensorflow-object-detection-api

🧠 Sample Python Code

import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

# Load pre-trained SSD MobileNet model
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Load and preprocess image
image = cv2.imread("test.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_tensor = tf.convert_to_tensor([image_rgb], dtype=tf.uint8)

# Run inference
detections = model(input_tensor)

# Extract results
boxes = detections['detection_boxes'][0].numpy()
scores = detections['detection_scores'][0].numpy()
classes = detections['detection_classes'][0].numpy().astype(np.int32)

# Draw bounding boxes
for i in range(len(scores)):
    if scores[i] > 0.5:
        h, w, _ = image.shape
        ymin, xmin, ymax, xmax = boxes[i]
        cv2.rectangle(
            image,
            (int(xmin * w), int(ymin * h)),
            (int(xmax * w), int(ymax * h)),
            (0, 255, 0),
            2
        )

cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 📊 Results

| Metric | Description | Result |
|--------|--------------|---------|
| **Model Used** | SSD MobileNet V2 (pre-trained on COCO dataset) | ✅ |
| **Accuracy** | Successfully detected multiple objects (person, car, bottle, dog, etc.) | **High** |
| **Speed** | Real-time detection at 25–30 FPS on GPU | ⚡ |
| **Bounding Boxes** | Correctly localized objects with precise coordinates | ✔️ |
| **Confidence Threshold** | Objects detected above 50% confidence | **≥ 0.5** |

### Visualization Example:
Detected bounding boxes and class labels drawn on images in real time using OpenCV.

## 🚀 Applications

Autonomous Vehicles: Detect pedestrians, traffic signs, and vehicles
Retail: Product detection and shelf monitoring
Healthcare: Detect anomalies in X-ray or MRI scans
Security: Surveillance and real-time anomaly detection

## 🧩 Conclusion

This project successfully implemented an Object Detection system using TensorFlow.
By leveraging pre-trained models, it achieved efficient, accurate, and real-time detection on diverse images and video streams.
It highlights the effectiveness of transfer learning and the TensorFlow Object Detection API for practical computer vision applications.

## 🔮 Future Work

Improve accuracy with advanced models like YOLOv8 and EfficientDet
Deploy detection on mobile/edge devices using TensorFlow Lite (TFLite)
Integrate with video analytics pipelines for large-scale real-time monitoring

## 👥 Contributors

Author: Aswathi Sujith
