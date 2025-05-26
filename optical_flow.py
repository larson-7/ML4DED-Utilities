import cv2
import numpy as np
import os

# The video feed is read in as
# a VideoCapture object
cap = cv2.VideoCapture("/Users/jordanlarson/engineering/cs8903/DEDWallVideos_Cropped/buildplate000_5.mp4")

# ret = a boolean return value from
# getting the frame, first_frame = the
# first frame in the entire video sequence
ret, first_frame = cap.read()

# Converts frame to grayscale because we
# only need the luminance channel for
# detecting edges - less computationally
# expensive
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Creates an image filled with zero
# intensities with the same dimensions
# as the frame
mask = np.zeros_like(first_frame)

# Sets image saturation to maximum
mask[..., 1] = 255

input_path = "/Users/jordanlarson/engineering/cs8903/DEDWallVideos_Cropped/buildplate000_5.mp4"
output_dir = "/Users/jordanlarson/engineering/cs8903/DEDWallVideosOpticalFlow"
output_file = "optical_flow_buildplate000_5.mp4"
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
frame_idx = 0
while True:
    # ret = a boolean return value from getting
    # the frame, frame = the current frame being
    # projected in the video
    ret, frame = cap.read()
    if not ret:
        break

    # Converts each frame to grayscale - we previously
    # only converted the first frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculates dense optical flow by Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                       None,
                                       0.5, 3, 15, 3, 5, 1.2, 0)

    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Sets image hue according to the optical flow
    # direction
    mask[..., 0] = angle * 180 / np.pi / 2

    # Sets image value according to the optical flow
    # magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    out.write(rgb)
    # Opens a new window and displays the output frame
    cv2.imshow("dense optical flow", rgb)

    # Updates previous frame
    prev_gray = gray
    frame_idx += 1
    print(f"Processed frame {frame_idx}/{total_frames}")

# The following frees up resources and
# closes all windows
cap.release()
out.release()
cv2.destroyAllWindows()
print("Depth video saved.")