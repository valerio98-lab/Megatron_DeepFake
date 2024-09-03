import os
import cv2
import dlib

# Paths
INPUT_VIDEO_PATH = "H:\\My Drive\\Megatron_DeepFake\\dataset"
OUTPUT_VIDEO_PATH = "H:\\My Drive\\Megatron_DeepFake\\dataset_processed"

# Ensure the output directory exists
if not os.path.exists(OUTPUT_VIDEO_PATH):
    os.makedirs(OUTPUT_VIDEO_PATH)

# Collect video files
VIDEO_FILES = []
for root, _, files in os.walk(INPUT_VIDEO_PATH):
    for file in files:
        if file.endswith(".mp4"):
            full_path = os.path.join(root, file)
            if os.path.exists(full_path):  # Check if the file exists
                VIDEO_FILES.append(full_path)

# Face detection and video codec
FACE_DETECTOR = dlib.get_frontal_face_detector()
CODEC = cv2.VideoWriter_fourcc(*"mp4v")

# File handling
with open("./BLACKLIST.txt", "a", encoding="utf-8") as blacklist, open(
    "./PROCESSED.txt", "a", encoding="utf-8"
) as processed:
    for in_video in VIDEO_FILES:
        cap = cv2.VideoCapture(in_video)
        if not cap.isOpened():
            blacklist.write(in_video + "\n")
            cap.release()
            continue

        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Build output file path
        output_file_path = os.path.join(
            OUTPUT_VIDEO_PATH, os.path.relpath(in_video, INPUT_VIDEO_PATH)
        )
        out_video = cv2.VideoWriter(
            output_file_path, CODEC, fps, (frame_width, frame_height)
        )

        # Read and process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if not FACE_DETECTOR(frame):  # Check for no faces detected
                continue
            out_video.write(frame)

        # Clean up
        processed.write(in_video + "\n")
        out_video.release()
        cap.release()
