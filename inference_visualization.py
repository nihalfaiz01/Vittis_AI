import torch
import numpy as np
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dataset import parse_bbox_file, voxelize
from model import Simple3DCNN
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# -----------------------------
# Paths
# -----------------------------
pc_file = r"C:\Users\nihal.faiz\Desktop\3D_point_cloud_project\warehouse_dataset\test\pointclouds\000411.bin"
label_file = r"C:\Users\nihal.faiz\Desktop\3D_point_cloud_project\warehouse_dataset\test\labels\000411.txt"

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

    print(f"Actual: {bbox['label']:15} → Predicted: {pred_label:15} "
          f"(Confidence: {100*pred_conf:.2f}%)")

# -----------------------------
# Visualization with Matplotlib
# -----------------------------

# Assign unique colors to each class
unique_classes = le.classes_
cmap = cm.get_cmap("tab10", len(unique_classes))
class_to_color = {cls: mcolors.rgb2hex(cmap(i)) for i, cls in enumerate(unique_classes)}

def draw_bbox(ax, bbox, color="r", linewidth=2, alpha=0.2):
    """Draw 3D bounding box with solid fill and label text."""
    x_min, x_max = bbox['x_min'], bbox['x_max']
    y_min, y_max = bbox['y_min'], bbox['y_max']
    z_min, z_max = bbox['z_min'], bbox['z_max']

    # 8 corners of the bounding box
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

    # Define the 6 faces of the box
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],  # bottom
        [corners[4], corners[5], corners[6], corners[7]],  # top
        [corners[0], corners[1], corners[5], corners[4]],  # front
        [corners[2], corners[3], corners[7], corners[6]],  # back
        [corners[1], corners[2], corners[6], corners[5]],  # right
        [corners[4], corners[7], corners[3], corners[0]]   # left
    ]

    # Add solid faces
    face_collection = Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors=color, alpha=alpha)
    ax.add_collection3d(face_collection)

    # Draw edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    for e in edges:
        ax.plot(*zip(corners[e[0]], corners[e[1]]), color=color, linewidth=linewidth)

    # Add label + confidence
    label_text = f"{bbox['pred_label']} ({100*bbox['confidence']:.1f}%)"
    cx, cy, cz = corners[6]  # top-right-front
    ax.text(cx, cy, cz, label_text, color="black", fontsize=8, backgroundcolor="white")

# -----------------------------
# Plotting
# -----------------------------
fig = plt.figure(figsize=(14, 6))

# Top-down view
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=pc[:, 2], cmap='viridis', s=1)
for bbox in bboxes:
    color = class_to_color.get(bbox["pred_label"], "black")
    draw_bbox(ax1, bbox, color=color, alpha=0.3)
ax1.view_init(elev=90, azim=-90)
ax1.set_title("Top-Down View")

# Perspective view
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=pc[:, 2], cmap='viridis', s=1)
for bbox in bboxes:
    color = class_to_color.get(bbox["pred_label"], "black")
    draw_bbox(ax2, bbox, color=color, alpha=0.3)
ax2.view_init(elev=20, azim=-60)
ax2.set_title("Perspective View")

# Legend → only show predicted classes
predicted_classes = set(b["pred_label"] for b in bboxes if b["pred_label"] != "None")
for cls in predicted_classes:
    color = class_to_color.get(cls, "black")
    ax1.plot([], [], [], color=color, label=cls)
ax1.legend(loc="upper right", fontsize=8, title="Predicted Classes")

plt.tight_layout()
plt.show()
