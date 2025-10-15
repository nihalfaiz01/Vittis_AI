import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import joblib
from collections import Counter
from dataset import load_dataset, PointCloudDataset
from model import Simple3DCNN
import os

# -----------------------------
# Paths
# -----------------------------
train_pc_dir = r"C:\Users\nihal.faiz\Desktop\Vitis_project\warehouse_dataset\train\pointclouds"
train_label_dir = r"C:\Users\nihal.faiz\Desktop\Vitis_project\warehouse_dataset\train\labels"
val_pc_dir = r"C:\Users\nihal.faiz\Desktop\Vitis_project\warehouse_dataset\val\pointclouds"
val_label_dir = r"C:\Users\nihal.faiz\Desktop\Vitis_project\warehouse_dataset\val\labels"

# -----------------------------
# Load Data
# -----------------------------
train_bboxes = load_dataset(train_pc_dir, train_label_dir)
val_bboxes   = load_dataset(val_pc_dir, val_label_dir)

print("Train classes:", Counter([b['label'] for b in train_bboxes]))

all_labels = [b['label'] for b in train_bboxes + val_bboxes]
le = LabelEncoder()
le.fit(all_labels)
num_classes = len(le.classes_)

train_dataset = PointCloudDataset(train_bboxes, le)
val_dataset   = PointCloudDataset(val_bboxes, le)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8)

# -----------------------------
# Model setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Simple3DCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# Training Loop
# -----------------------------
EPOCHS = 15
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for voxels, labels in train_loader:
        voxels, labels = voxels.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(voxels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# -----------------------------
# Save Model & Label Encoder
# -----------------------------
os.makedirs("saved_models", exist_ok=True)

torch.save(model.state_dict(), "saved_models/cnn_model.pth")
print("Model saved at saved_models/cnn_model.pth")

joblib.dump(le, "saved_models/label_encoder.pkl")
print("LabelEncoder saved at saved_models/label_encoder.pkl")
