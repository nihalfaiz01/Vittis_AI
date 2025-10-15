import torch
import numpy as np
import joblib
from dataset import parse_bbox_file, voxelize
from model import Simple3DCNN

# -----------------------------
# Paths
# -----------------------------
pc_file = r"C:\Users\nihal.faiz\Desktop\Vitis_project\warehouse_dataset\test\pointclouds\000411.bin"
label_file = r"C:\Users\nihal.faiz\Desktop\Vitis_project\warehouse_dataset\test\labels\000411.txt"

model_path = "saved_models/cnn_model.pth"
label_encoder_path = "saved_models/label_encoder.pkl"

# -----------------------------
# Load LabelEncoder and Model
# -----------------------------
le = joblib.load(label_encoder_path)
num_classes = len(le.classes_)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Simple3DCNN(num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# -----------------------------
# Load point cloud + bounding boxes
# -----------------------------
data = np.fromfile(pc_file, dtype=np.float32)
num_features = 3 if data.size % 3 == 0 else 4
pc = data.reshape(-1, num_features)[:, :3]

bboxes = parse_bbox_file(label_file)

# -----------------------------
# Run inference on each bounding box
# -----------------------------
softmax = torch.nn.Softmax(dim=1)

for bbox in bboxes:
    mask = (
        (pc[:, 0] >= bbox['x_min']) & (pc[:, 0] <= bbox['x_max']) &
        (pc[:, 1] >= bbox['y_min']) & (pc[:, 1] <= bbox['y_max']) &
        (pc[:, 2] >= bbox['z_min']) & (pc[:, 2] <= bbox['z_max'])
    )
    points_inside = pc[mask]

    if points_inside.shape[0] == 0:
        print(f"⚠️ No points inside bounding box {bbox['label']}")
        continue

    voxel = voxelize(points_inside, grid_size=32)
    voxel_tensor = torch.tensor(voxel, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(voxel_tensor)
        probs = softmax(output)
        pred_class = probs.argmax(1).cpu().item()
        pred_conf = probs[0, pred_class].cpu().item()
        pred_label = le.inverse_transform([pred_class])[0]

    print(f"actual: {bbox['label']:15} → Predicted: {pred_label:15} "
          f"(Confidence Score : {100*pred_conf:.2f} %)")
