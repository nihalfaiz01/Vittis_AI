import numpy as np
import open3d as o3d
import os

# ---------------------------
# User settings
# ---------------------------
bin_file = r"C:\Users\nihal.faiz\Desktop\centerPoint\warehouse_dataset\test\pointclouds\000411.bin"
label_file = r"C:\Users\nihal.faiz\Desktop\centerPoint\warehouse_dataset\test\labels\000411.txt"

# ---------------------------
# Load LiDAR points
# ---------------------------
points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
print(f"Number of points: {points.shape[0]}")
print(f"First 5 points:\n{points[:5]}")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])

# Color points blue
colors = np.tile(np.array([[0, 0, 1]]), (points.shape[0], 1))
pcd.colors = o3d.utility.Vector3dVector(colors)

# ---------------------------
# Add coordinate frame
# ---------------------------
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

# ---------------------------
# Load bounding boxes
# ---------------------------
bbox_objs = []

if os.path.exists(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.strip().split()
            if len(tokens) != 8:
                continue
            cls = tokens[0]
            x, y, z, dx, dy, dz, yaw = map(float, tokens[1:])
            
            # Convert yaw to radians if needed
            # yaw = np.deg2rad(yaw)  # uncomment if yaw is in degrees
            
            center = np.array([x, y, z])
            size = np.array([dx, dy, dz])
            
            # Create oriented bounding box
            R = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, yaw])
            obb = o3d.geometry.OrientedBoundingBox(center, R, size)
            obb.color = (1, 0, 0)  # red
            bbox_objs.append(obb)
else:
    print(f"Label file not found: {label_file}")

# ---------------------------
# Visualize
# ---------------------------
o3d.visualization.draw_geometries([pcd, coord_frame, *bbox_objs],
                                  window_name="3D LiDAR Viewer with Labels",
                                  width=1280, height=720)
