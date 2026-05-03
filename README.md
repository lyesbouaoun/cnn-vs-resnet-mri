
# 🧠 CNN vs ResNet – MRI Classification


## 🚀 Project Overview

This project compares two deep learning architectures for **MRI image classification**:

- 🧠 Custom CNN model (baseline)
- 🔬 ResNet18 model (deep residual learning)

The goal is to evaluate performance differences in a medical imaging context.

---

## 🎯 Objectives

- Build a CNN baseline for MRI classification  
- Implement ResNet18 for performance comparison  
- Analyze training stability and convergence  
- Evaluate accuracy on medical images  
- Understand deep learning impact in healthcare AI  

---

## 🏗️ Project Structure
## 🏗️ Project Structure


cnn-vs-resnet-mri/
│
├── cnn_model.py
├── cnn_train.py
├── resnet18_model.py
├── resnet_train.py
├── main.py
├── config.py
├── util_dataset.py
├── dataset_mean_std.py
├── affichage.py
├── exigence.txt
└── README.md


---

## 🧠 Models

### CNN (Baseline)
- Convolution + Pooling layers
- Fully connected classifier
- Fast but limited depth

### ResNet18
- Residual connections (skip connections)
- Solves vanishing gradient problem
- Better performance on complex MRI data

---

## 📊 Results

- CNN: good baseline performance
- ResNet18: higher accuracy and better generalization
- ResNet shows more stable convergence

📉 Loss curves  
📈 Accuracy curves  
⚖️ Model comparison graphs

(Generated via `affichage.py`)

---

## ⚙️ Installation

```bash
git clone https://github.com/lyesbouaoun/cnn-vs-resnet-mri
cd cnn-vs-resnet-mri
pip install -r exigence.txt

## Train CNN
python cnn_train.py

## Train ResNet18
python resnet_train.py

▶️ Run Full Pipeline
python main.py

⚙️ Configuration

Edit config.py:

learning rate
batch size
epochs
device (CPU/GPU)
dataset path
🧪 Data Processing
Image resizing
Tensor conversion
Normalization (mean/std)
Dataset loader in util_dataset.py
🏥 Medical AI Impact

This project applies deep learning to medical imaging for:

Early disease detection
MRI classification support
AI-assisted diagnosis systems
💼 Skills Demonstrated
Deep Learning (CNN, ResNet)
PyTorch implementation
Medical image processing
Model comparison & evaluation
Training pipeline design
Data preprocessing & normalization
Visualization of training metrics
📈 Key Insight

ResNet18 outperforms CNN due to:

Deeper architecture
Residual connections
Better gradient flow
Improved generalization on MRI data
👨‍💻 Author

Lyes BOUAOUN

Machine Learning / Deep Learning Engineer
Computer Vision & Medical AI
Actively seeking AI / Computer Vision internship

