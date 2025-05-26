import cv2
import torch
import os
from DepthAnything.depth_anything_v2.dpt import DepthAnythingV2
import numpy as np


def depth_to_colormap(depth_map: np.ndarray, colormap=cv2.COLORMAP_INFERNO) -> np.ndarray:
    """
    Convert a float depth map to a color-mapped BGR image.

    Args:
        depth_map (np.ndarray): Float32/64 depth map of shape (H, W)
        colormap (int): OpenCV colormap (e.g., cv2.COLORMAP_INFERNO)

    Returns:
        np.ndarray: BGR image of shape (H, W, 3), dtype=uint8
    """
    # Normalize depth to 0–255
    depth_min = np.min(depth_map)
    depth_max = np.max(depth_map)

    # Avoid division by zero
    if depth_max - depth_min < 1e-6:
        depth_normalized = np.zeros_like(depth_map, dtype=np.uint8)
    else:
        depth_normalized = 255 * (depth_map - depth_min) / (depth_max - depth_min)
        depth_normalized = depth_normalized.astype(np.uint8)

    # Apply color map
    depth_colored = cv2.applyColorMap(depth_normalized, colormap)
    return depth_colored

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}

encoder = "vitb"  # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(
    torch.load(
        f"DepthAnything/checkpoints/depth_anything_v2_{encoder}.pth", map_location="cpu"
    )
)
model = model.to(DEVICE).eval()

# Paths
input_path = "/Users/jordanlarson/engineering/cs8903/DEDWallVideos_Cropped/buildplate000_5.mp4"
output_dir = "/Users/jordanlarson/engineering/cs8903/DEDWallVideosDepth"
output_file = "depth_buildplate000_5.mp4"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_file)

# Open input video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error: Cannot open input video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create video writer for depth output (single channel)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames in video: {total_frames}")

# Frame loop
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # forward pass of model
    depth_np = model.infer_image(frame)
    depth_colormap = depth_to_colormap(depth_np)
    # write depth frame
    out.write(depth_colormap)

    frame_idx += 1
    print(f"Processed frame {frame_idx}/{total_frames}")

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print("Depth video saved.")