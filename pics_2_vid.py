import os
import cv2
from glob import glob
from natsort import natsorted

def make_video_from_images(image_folder, output_path, fps=10):
    images = natsorted(glob(os.path.join(image_folder, "*.jpg")))
    if not images:
        print(f"No images found in {image_folder}")
        return

    # Read first image to get dimensions
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape

    video_writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    for img_path in images:
        frame = cv2.imread(img_path)
        video_writer.write(frame)

    video_writer.release()
    print(f"Saved video: {output_path}")

def process_all_subfolders(base_dir, output_dir="videos", fps=10):
    os.makedirs(output_dir, exist_ok=True)
    for entry in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, entry)
        if os.path.isdir(subdir_path):
            output_video_path = os.path.join(output_dir, f"{entry}.mp4")
            make_video_from_images(subdir_path, output_video_path, fps=fps)

if __name__ == "__main__":
    base_directory = "/Users/jordanlarson/OMSCS/OMSCS-Courses/cs8903/layer_images"
    process_all_subfolders(base_directory)
