# Multi-Modal Deception Detection using Real-Life Trials Videos

This project aims to develop a model for multi-modal deception detection using real-life trial videos. The model leverages advanced video processing techniques and a fine-tuned 3D ResNet-50 architecture to detect deceptive behavior.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)

## Introduction
This project involves the classification of deception in real-life trial videos. We use a 3D ResNet-50 model pre-trained on the Kinetics-400 dataset and fine-tune it for our specific task. The dataset comprises 120 videos, which are preprocessed and split into training and test sets.

## Dataset
- **Number of videos**: 120
- **Video segmentation**: Each video is divided into 1.5-second subvideos.
- **Frames extraction**: 30 frames are extracted from each subvideo and combined to create 3D image-like data.

## Preprocessing
1. **Face Extraction**:
   - Initial extraction using CascadeClassifier (later replaced by RetinaFace for better accuracy).
2. **Data Resizing**:
   - Original dimensions: 128x128
   - Resized to: 224x224 to match the Kinetics-400 dataset.
3. **Zero Padding**:
   - Added to the beginning and end of the time/frame dimension to ensure compatibility with the model.

## Model Architecture
- **Base Model**: 3D ResNet-50
- **Pre-trained on**: Kinetics-400 dataset
- **Fine-tuning**: Adjusted to enhance performance on our specific dataset.

## Training
- **Batch Size**: 8
- **Hardware**: A100 GPU with high RAM on Google Colab.
- **Challenges**:
  - Managing memory and size issues due to 3D images.
  - Ensuring efficient data handling and processing.
