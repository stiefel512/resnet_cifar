# Training ResNet from Scratch on CIFAR10

## Overview
This project implements and trains a ResNet-style Convolutional Neural Network **from scratch**. 
The goal is not to achieve SOTA accuracy, but to demonstrate a clear understanding of:

- Training dynamics and optimization
- Regularization and overfitting
- Reproducibility
- Failure modes and diagnostics

All experiments are run without pretrained weights.

---

## Problem Statement
CIFAR-10 is a standard image classification benchmark consisting of 60,000 RGB images (32x32) across 10 classes

Despite its simplicity, CIFAR-10 exposes several important challenges
- Limited spatial resolution
- High intra-class variance
- Fast overfitting with modern architectures

This makes it a good testbed for analyzing training behavior rather than just final accuracy.

---

## Dataset
- **Source:** CIFAR-10 (Torchvision)
- **Classes:** 10
- **Train/Validation Split:** 40,000 / 10,000
- **Test Set:** Official 10,000 images

### Data Augmentations
Applied only to the training set
- Random crop with padding (4 pixels)
- Random horizontal flip

Validation and test sets are kept deterministic

---

## Model Architecture
A custom ResNet-18–style architecture adapted for CIFAR-10:
- No initial 7x7 convolution or max-pooling
- First convolution: 3x3, stride 1
- Batch or Group Normalization after every convolution
- ReLU activations
- Global Average Pooling before classification

All convolutional and linear layers were initialized using Kaiming normal initialization for ReLU networks; normalization layers were initialized to identity. 

---

## Training Setup

### Optimization
- **Optimizer:** SGD with momentum (0.9)
- **Initial Learning Rate:** 0.1
- **Weight Decay:** 5e-4
- **Batch Size:** 128

### Learning Rate Schedule
- Cosine Annealing (eta_min 1e-6) with 5 epochs warmup
- Total training: 200 epochs

### Loss Function
- Cross-entropy loss

---

## Reproducibility
To ensure deterministic behavior:
- Python, NumPy, and PyTorch seeds are fixed
- CuDNN deterministic mode enabled
- Train/validation split uses a fixed generator

Note: Full determinism is hardware-dependent and small variations may still occur.

---

## Results

| Metric        | Value |
|---------------|-------|
| Train Accuracy | ~99.98% |
| Val Accuracy   | ~94.77% |
| Test Accuracy  | ~93.64% |

### Learning Curves

As we can see, with regularization the model never overfits. The model validation loss keeps descending and the validation accuracy keeps rising throughout the training.

![Loss Curve](figures/train_val_loss.png)
![Accuracy Curve](figures/train_val_accuracy.png)

---

## Failure Analysis
Common failure cases include:
- Confusion between visually similar classes (e.g., cat vs dog)
- Sensitivity to background texture
- Poor calibration on ambiguous samples

A gallery of misclassified examples is included in `figures/failures/`.

---

## Key Observations

- Training from scratch on CIFAR-10 is highly sensitive to learning rate choice.
- Increasing depth beyond ResNet-18 yields diminishing returns without stronger regularization.

---

## Project Structure

```text
├── configs/
│ └── config.yaml
├── data/
├── experiments/
├── figures/
├── model.py
├── README.md
├── requirements.txt
├── test.py
└── train.py
```

---

## How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train

```bash
python train.py --config configs/config.yaml
```

### Evaluate

```bash
python test.tpy --config configs/config.yaml
```

---

## What I Would Do Next

- Add label smoothing and compare calibration
- Experiment with cosine learning rate scheduling
- Compare against a ViT-style architecture under identical conditions
- Extend analysis to CIFAR-100 to study class imbalance effects

---

## Takeaway

This project focuses on understanding training behavior, not leaderboard performance.
It serves as a controlled environment for studying optimization, regularization, and model diagnostics in convolutional neural networks.
