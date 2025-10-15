import torch
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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
        bbox["pred_label"] = "None"
        bbox["confidence"] = 0.0
        print(f"GT: {bbox['label']:12} → Pred: None ❌ (no points inside box)")
        continue

    voxel = voxelize(points_inside, grid_size=32)
    voxel_tensor = torch.tensor(voxel, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(voxel_tensor)
        probs = softmax(output)
        pred_class = probs.argmax(1).cpu().item()
        pred_conf = probs[0, pred_class].cpu().item()
        pred_label = le.inverse_transform([pred_class])[0]

    bbox["pred_label"] = pred_label
    bbox["confidence"] = pred_conf

    match_symbol = "✅" if pred_label == bbox['label'] else "❌"
    print(f"GT: {bbox['label']:12} → Pred: {pred_label:12} "
          f"(Conf: {100*pred_conf:.1f}%) {match_symbol}")

# -----------------------------
# Visualization
# -----------------------------
def draw_bbox(ax, bbox, color="r", linewidth=2, label_text=None):
    """Draw 3D bounding box."""
    x_min, x_max = bbox['x_min'], bbox['x_max']
    y_min, y_max = bbox['y_min'], bbox['y_max']
    z_min, z_max = bbox['z_min'], bbox['z_max']

    corners = np.array([
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max]
    ])

    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    # Draw edges
    for e in edges:
        ax.plot(*zip(corners[e[0]], corners[e[1]]), color=color, linewidth=linewidth)

    # Add label
    if label_text:
        cx, cy, cz = corners[6]
        ax.text(cx, cy, cz, label_text, color=color, fontsize=8, backgroundcolor="white")

# -----------------------------
# Plotting
# -----------------------------
fig = plt.figure(figsize=(14, 6))

# Top-down view
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=pc[:, 2], cmap='viridis', s=1)

for bbox in bboxes:
    # Ground Truth → black wireframe
    draw_bbox(ax1, bbox, color="black", linewidth=1,
              label_text=f"GT: {bbox['label']}")

    # Prediction → solid color
    pred_text = f"{bbox['pred_label']} ({100*bbox['confidence']:.1f}%)"
    color = "blue" if bbox['pred_label'] == "Box" else "green"
    draw_bbox(ax1, bbox, color=color, linewidth=2, label_text=pred_text)

ax1.view_init(elev=90, azim=-90)
ax1.set_title("Top-Down View")

# Perspective view
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=pc[:, 2], cmap='viridis', s=1)

for bbox in bboxes:
    draw_bbox(ax2, bbox, color="black", linewidth=1,
              label_text=f"GT: {bbox['label']}")

    pred_text = f"{bbox['pred_label']} ({100*bbox['confidence']:.1f}%)"
    color = "blue" if bbox['pred_label'] == "Box" else "green"
    draw_bbox(ax2, bbox, color=color, linewidth=2, label_text=pred_text)

ax2.view_init(elev=20, azim=-60)
ax2.set_title("Perspective View")

plt.tight_layout()
plt.show()
