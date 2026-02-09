# 🔍 Steel Surface Defect Detection using YOLOv8

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Real-time detection of steel surface defects for autonomous building-washing drone inspection system**

[Overview](#overview) • [Results](#results) • [Installation](#installation) • [Usage](#usage) • [Dataset](#dataset) • [Model Comparison](#model-comparison)

</div>

---

## 📋 Overview

This project implements a deep learning-based defect detection system using YOLOv8 for identifying surface defects on steel structures. The system is designed to be deployed on autonomous drones for building facade inspection before cleaning operations.

### 🎯 Key Features

- **Real-time Detection**: 4.2ms inference time per image
- **High Accuracy**: 74.8% mAP@0.5
- **6 Defect Classes**: Comprehensive coverage of common steel defects
- **Production Ready**: Optimized for edge device deployment

### 🔍 Detected Defect Types

| ID | Defect Type | Description |
|----|-------------|-------------|
| 0 | Crazing | Fine crack patterns on the surface |
| 1 | Inclusion | Foreign material embedded in steel |
| 2 | Patches | Surface irregularities and discoloration |
| 3 | Pitted Surface | Small holes or pits on the surface |
| 4 | Rolled-in Scale | Scale pressed into surface during rolling |
| 5 | Scratches | Linear marks on the surface |

---

## 📊 Results

### Overall Performance

| Model | mAP@0.5 | mAP@0.5:0.95 | Parameters | Inference Time |
|-------|---------|--------------|------------|----------------|
| **YOLOv8n** | **0.748** | **0.402** | 3.0M | 4.2ms ✅ |
| YOLOv8s | 0.739 | 0.390 | 11.1M | 9.8ms |

### Per-Class Performance (YOLOv8n)

| Class | Precision | Recall | mAP@0.5 |
|-------|-----------|--------|---------|
| Crazing | 0.664 | 0.333 | 0.519 |
| Inclusion | 0.808 | 0.717 | 0.820 |
| **Patches** | **0.863** | **0.824** | **0.921** |
| Pitted Surface | 0.779 | 0.736 | 0.836 |
| Rolled-in Scale | 0.597 | 0.449 | 0.561 |
| Scratches | 0.659 | 0.901 | 0.830 |

### Training Curves

<div align="center">
<img src="results/results_comparison.png" width="80%">
</div>

---

## 🔬 Model Comparison Analysis

An interesting finding from this project:

> **The smaller model (YOLOv8n) outperformed the larger model (YOLOv8s)!**

### Why?

1. **Dataset Size**: With only 1,800 training images, larger models tend to overfit
2. **Generalization**: YOLOv8n provides better generalization for this specific task
3. **Efficiency**: Smaller model = faster inference = better for real-time drone applications

### Lesson Learned
**Bigger isn't always better** — model size should match your dataset size and use case requirements.

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/steel-defect-detection.git
cd steel-defect-detection

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Training

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='data/data.yaml',
    epochs=150,
    imgsz=640,
    batch=32,
    device=0,
    patience=50
)
```

### Inference

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('weights/best.pt')

# Run inference
results = model.predict(
    source='path/to/image.jpg',
    conf=0.5,
    save=True
)
```

### Data Configuration

```yaml
# data.yaml
path: /path/to/dataset
train: images/train
val: images/val

nc: 6
names: ['crazing', 'inclusion', 'patches', 
        'pitted_surface', 'rolled-in_scale', 'scratches']
```

---

## 📁 Dataset

### NEU-DET Dataset

| Property | Value |
|----------|-------|
| **Source** | [Northeastern University Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) |
| **Total Images** | 1,800 grayscale images |
| **Image Size** | 200 × 200 pixels |
| **Classes** | 6 defect types |
| **Train/Val Split** | 80% / 20% |

### Preprocessing

1. Converted XML annotations (Pascal VOC) to YOLO format
2. Images resized to 640×640 during training
3. Applied data augmentation (Mosaic, flip, HSV adjustments)

---

## 📂 Project Structure

```
steel-defect-detection/
├── data/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml
├── weights/
│   ├── yolov8n_best.pt
│   └── yolov8s_best.pt
├── results/
│   ├── confusion_matrix.png
│   ├── results.png
│   └── results.csv
├── notebooks/
│   └── training.ipynb
├── scripts/
│   ├── train.py
│   ├── predict.py
│   └── convert_annotations.py
├── requirements.txt
└── README.md
```

---

## 🔧 Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 150 |
| Image Size | 640×640 |
| Batch Size | 32 (YOLOv8n) / 16 (YOLOv8s) |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Early Stopping | 50 epochs |
| GPU | Tesla T4 (15GB) |

---

## 🎯 Future Work

- [ ] Deploy model on Jetson Nano for drone integration
- [ ] Add more defect classes
- [ ] Implement model quantization for faster inference
- [ ] Create web demo using Gradio/Streamlit
- [ ] Train on larger dataset for better generalization

---

## 📚 References

1. Meng, J., & Wen, S. (2024). "Detection of Steel Surface Defects Based on Improved YOLOv8n Algorithm." *2024 International Conference on AI-Powered Medical and Vocational Practice (AIPMV)*, IEEE.

2. [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)

3. [NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Mohammed Abdulqawi Alezzi Saleh**

- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/YOUR_PROFILE)

---

## ⭐ Show Your Support

Give a ⭐ if this project helped you!

---

<div align="center">
Made with ❤️ for autonomous drone inspection systems
</div>
