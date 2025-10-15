import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

# -----------------------------
# Parse bounding box labels (center → min/max)
# -----------------------------
def parse_bbox_file(file_path):
    """
    Parse a label file.
    Format: class cx cy cz length width height yaw
    Converts to axis-aligned min/max box.
    """
    bboxes = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:  # need 7 numbers + label
                continue
            try:
                label = parts[0]
                cx, cy, cz = map(float, parts[1:4])   # center coordinates
                l, w, h = map(float, parts[4:7])     # box size
                # yaw = float(parts[7])               # rotation (ignored for now)

                # Convert center+size → min/max (axis-aligned)
                x_min, x_max = cx - l/2, cx + l/2
                y_min, y_max = cy - w/2, cy + w/2
                z_min, z_max = cz - h/2, cz + h/2

                bboxes.append({
                    "label": label,
                    "x_min": x_min, "x_max": x_max,
                    "y_min": y_min, "y_max": y_max,
                    "z_min": z_min, "z_max": z_max
                })
            except ValueError:
                continue
    return bboxes

# -----------------------------
# Load dataset
# -----------------------------
def load_dataset(pc_dir, label_dir):
    """
    Load point clouds and attach bounding box points.
    """
    pc_files = sorted(glob.glob(os.path.join(pc_dir, "*.bin")))
    label_files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
    all_bboxes = []

    for pc_file, label_file in zip(pc_files, label_files):
        data = np.fromfile(pc_file, dtype=np.float32)
        num_features = 3 if data.size % 3 == 0 else 4
        pc = data.reshape(-1, num_features)[:, :3]

        bboxes = parse_bbox_file(label_file)
        for bbox in bboxes:
            mask = (
                (pc[:, 0] >= bbox["x_min"]) & (pc[:, 0] <= bbox["x_max"]) &
                (pc[:, 1] >= bbox["y_min"]) & (pc[:, 1] <= bbox["y_max"]) &
                (pc[:, 2] >= bbox["z_min"]) & (pc[:, 2] <= bbox["z_max"])
            )
            points_inside = pc[mask]
            if points_inside.shape[0] > 0:
                bbox["points"] = points_inside
                all_bboxes.append(bbox)
    return all_bboxes

# -----------------------------
# Voxelization
# -----------------------------
def voxelize(points, grid_size=32):
    """
    Convert point cloud (N x 3) into 3D voxel grid.
    """
    if len(points) == 0:
        return np.zeros((1, grid_size, grid_size, grid_size), dtype=np.float32)

    pts_min = points.min(0)
    pts_max = points.max(0)
    pts_norm = (points - pts_min) / (pts_max - pts_min + 1e-6)
    pts_idx = np.clip((pts_norm * (grid_size - 1)).astype(int), 0, grid_size - 1)

    voxels = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    voxels[pts_idx[:, 0], pts_idx[:, 1], pts_idx[:, 2]] = 1.0
    return voxels[np.newaxis, :, :, :]  # add channel dimension

# -----------------------------
# PyTorch Dataset
# -----------------------------
class PointCloudDataset(Dataset):
    def __init__(self, bboxes, label_encoder, grid_size=32):
        self.bboxes = bboxes
        self.le = label_encoder
        self.grid_size = grid_size

    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, idx):
        bbox = self.bboxes[idx]
        voxel = voxelize(bbox["points"], self.grid_size)
        label = self.le.transform([bbox["label"]])[0]
        return torch.tensor(voxel, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
