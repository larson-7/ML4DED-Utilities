import cv2
import torch
import matplotlib.pyplot as plt

from DepthAnything.depth_anything_v2.dpt import DepthAnythingV2

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

raw_img = cv2.imread("example.png")
depth_np = model.infer_image(raw_img)

# normalize for visualization
depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())

# Plot
plt.figure(figsize=(10, 6))
plt.imshow(depth_norm, cmap="plasma")
plt.colorbar(label="Normalized Depth")
plt.title("Predicted Depth Map")
plt.axis("off")
plt.savefig("example_depth.png")
