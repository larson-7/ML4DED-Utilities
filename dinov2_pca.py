import torch
import torchvision.transforms as T
import cv2
import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

# Parameters
patch_h, patch_w = 40, 40
feat_dim = 1536  # for vitg14
resize_h, resize_w = patch_h * 14, patch_w * 14

# Transform
transform = T.Compose([
    T.GaussianBlur(9, sigma=(0.1, 2.0)),
    T.Resize((resize_h, resize_w)),
    T.CenterCrop((resize_h, resize_w)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])

# Load model
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
device = torch.device("cuda")
dinov2 = dinov2.to(device)
dinov2.eval()

# Video paths
input_path = "/home/jordan/omscs/ML4DED/data/DEDWallVideos_Cropped/buildplate000_5.mp4"
output_dir = "/home/jordan/omscs/ML4DED/data/DEDWallVideos_Dino/DEDWallVideos_dino"
output_file = "dinov2_buildplate000_5.mp4"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_file)

# Setup video capture and writer
cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out = cv2.VideoWriter(output_path,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps,
                      (resize_w, resize_h))

# Pre-sample for PCA fitting
fit_samples = []
fit_limit = 50

print("Fitting PCA from first few frames...")
while len(fit_samples) < fit_limit:
    ret, frame = cap.read()
    if not ret:
        break
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = dinov2.forward_features(tensor)['x_norm_patchtokens'][0]  # (N, D)
    fit_samples.append(feat.cpu().numpy())
fit_features = np.concatenate(fit_samples, axis=0)
pca = PCA(n_components=3)
pca.fit(fit_features)

# Reset video stream
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Process video frame-by-frame
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret or frame_idx > 100:
        break
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = dinov2.forward_features(tensor)['x_norm_patchtokens'][0].cpu().numpy()
    transformed = pca.transform(feat)
    transformed = transformed.reshape(patch_h, patch_w, 3)

    # Normalize and convert to uint8
    vis = (transformed - transformed.min()) / (transformed.max() - transformed.min() + 1e-8)
    vis = (vis * 255).astype(np.uint8)
    vis = cv2.resize(vis, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    out.write(vis)
    frame_idx += 1
    print(f"Processed frame {frame_idx}/{total_frames}")

cap.release()
out.release()
cv2.destroyAllWindows()
print("PCA visualization video saved.")
