# Vittis_AI
3D Object Detection in Warehouse Environments using LiDAR & Vitis AI
Table of Contents

Project Overview

Dataset

Project Structure

Dependencies

Data Preprocessing

Training

Inference

Visualization

Vitis AI Deployment

Results

Future Work

Project Overview

This project implements a 3D object detection pipeline for warehouse environments using LiDAR point clouds. Key features include:

Bounding box extraction from point clouds

Voxelization of point clouds for 3D CNN input

Simple3DCNN model for object classification

Visualization using Matplotlib and Open3D

Deployment on FPGA using Vitis AI for real-time inference

Dataset

Dataset: Warehouse LiDAR Dataset (2024 SAE Paper)

File Types:

*.bin → LiDAR point clouds (x, y, z, intensity)

*.txt → Bounding box labels:

class cx cy cz length width height yaw


Directory Structure:

warehouse_dataset/
├── train/
│   ├── pointclouds/
│   └── labels/
├── val/
│   ├── pointclouds/
│   └── labels/
└── test/
    ├── pointclouds/
    └── labels/

Project Structure
project/
├── dataset.py            # Dataset parsing, voxelization, PointCloudDataset
├── model.py              # Simple3DCNN model definition
├── train.py              # Training script
├── test.py               # Inference and evaluation script
├── visualize.py          # Visualization with Matplotlib/Open3D
├── saved_models/         # Trained model weights and LabelEncoder
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

Dependencies

Install required packages:

pip install torch torchvision torchaudio
pip install numpy matplotlib scikit-learn open3d joblib
pip install tqdm


For Vitis AI deployment: Vitis AI tools

Data Preprocessing

Parse bounding boxes (center → min/max coordinates)

Voxelization: Convert points inside each bounding box → 32×32×32 voxel grid

Example usage:

from dataset import parse_bbox_file, voxelize

bboxes = parse_bbox_file("test/labels/000411.txt")
voxel_grid = voxelize(bboxes[0]["points"], grid_size=32)

Training

Script: train.py

Steps:

Load training and validation point clouds and labels

Create PointCloudDataset and DataLoader

Train Simple3DCNN for 15 epochs

Save trained model and LabelEncoder:

saved_models/cnn_model.pth
saved_models/label_encoder.pkl


Run command:

python train.py

Inference

Script: test.py

Steps:

Load test point cloud and labels

Parse bounding boxes and voxelize

Run inference using trained CNN

Print predicted label, confidence, and compare with ground truth

Run command:

python test.py

Visualization

Script: visualize.py

Options:

Matplotlib: Top-down & perspective 3D plots

Open3D: Interactive 3D visualization

Color-code predicted classes vs ground truth

Example:

from visualize import draw_bbox

Vitis AI Deployment

Quantize trained PyTorch model (INT8/FP16) using Vitis AI Quantizer

Compile for FPGA using Vitis AI Compiler

Run inference on Kria/Versal FPGA board

Collect latency, FPS, and confidence scores

Visualization is same as in visualize.py

Results

Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

Visualization: Predicted bounding boxes overlaid on point clouds

Vitis AI Performance: Real-time FPGA inference achieved

Future Work

Predict object orientation (yaw)

Test advanced 3D detection models (PV-RCNN, CenterPoint)

Optimize FPGA resource usage and latency

Integrate with warehouse robotic pick-and-place systems
