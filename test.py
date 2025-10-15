import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from dataset import load_dataset, PointCloudDataset   # <- from your dataset.py
from model import Simple3DCNN                         # <- from your model.py

# -----------------------------
# Paths
# -----------------------------
test_pc_dir = r"C:\Users\nihal.faiz\Desktop\3D_point_cloud_project\warehouse_dataset\test\pointclouds"
test_label_dir = r"C:\Users\nihal.faiz\Desktop\3D_point_cloud_project\warehouse_dataset\test\labels"

model_path = r"C:\Users\nihal.faiz\Desktop\3D_point_cloud_project\saved_models\cnn_model.pth"
label_encoder_path = r"C:\Users\nihal.faiz\Desktop\3D_point_cloud_project\saved_models\label_encoder.pkl"

# -----------------------------
# Load Data
# -----------------------------
print("Loading test dataset...")
test_bboxes = load_dataset(test_pc_dir, test_label_dir)

le = joblib.load(label_encoder_path)
num_classes = len(le.classes_)

test_dataset = PointCloudDataset(test_bboxes, le)
test_loader  = DataLoader(test_dataset, batch_size=8)

# -----------------------------
# Load Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Simple3DCNN(num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# -----------------------------
# Evaluate
# -----------------------------
print("Running evaluation...")
all_preds, all_labels, all_confs = [], [], []

softmax = torch.nn.Softmax(dim=1)

with torch.no_grad():
    for voxels, labels in test_loader:
        voxels, labels = voxels.to(device), labels.to(device)
        outputs = model(voxels)
        probs = softmax(outputs)

        preds = probs.argmax(1).cpu().numpy()
        confs = probs.max(1).values.cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_confs.extend(confs)

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_confs = np.array(all_confs)

# -----------------------------
# Metrics
# -----------------------------
all_class_indices = list(range(len(le.classes_)))  # ensures all classes included

print("\n=== Classification Report ===")
print(classification_report(
    all_labels,
    all_preds,
    labels=all_class_indices,
    target_names=le.classes_,
    zero_division=0
))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(
    all_labels,
    all_preds,
    labels=all_class_indices
))
