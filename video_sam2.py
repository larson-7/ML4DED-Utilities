import cv2
import torch
import os
import numpy as np
from sam2.sam2_video_predictor import SAM2VideoPredictor

# Load model
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", DEVICE)
predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large").to(DEVICE)

# Paths
input_path = "/Users/jordanlarson/engineering/cs8903/DEDWallVideos_Cropped/buildplate000_5.mp4"
output_dir = "/Users/jordanlarson/engineering/cs8903/DEDWallVideosSAM"
output_file = "sam2_buildplate000_5.mp4"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_file)

# Get video properties
cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

# Load video into predictor state
with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
    state = predictor.init_state(input_path)

    # Add a box prompt on the center of the first frame (dummy example)
    center_x, center_y = width // 2, height // 2
    box_size = 100
    box_prompt = torch.tensor([[center_x - box_size, center_y - box_size,
                                center_x + box_size, center_y + box_size]],
                              dtype=torch.float32, device=DEVICE)

    frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, input_boxes=box_prompt)

    # Propagate masks across all frames
    frame_dict = {i: frame for i, frame in enumerate(state["frames"])}
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        frame = frame_dict[frame_idx].cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        frame = np.transpose(frame, (1, 2, 0))  # CHW → HWC

        overlay = frame.copy()
        for mask in masks:
            mask_np = mask.squeeze().cpu().numpy()
            overlay[mask_np > 0] = (255, 0, 0)

        blended = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        bgr_out = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        out.write(bgr_out)

        print(f"Processed frame {frame_idx + 1}/{total_frames}")

# Cleanup
out.release()
print("SAM2 video saved.")
