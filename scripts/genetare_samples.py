import os
import cv2
import dlib
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Paths
INPUT_VIDEO_PATH = "G:\\My Drive\\Megatron_DeepFake\\dataset"
OUTPUT_VIDEO_PATH = "G:\\My Drive\\Megatron_DeepFake\\dataset_processed"

# Ensure the output directory exists
if not os.path.exists(OUTPUT_VIDEO_PATH):
    os.makedirs(OUTPUT_VIDEO_PATH)

# Initialize sets to store already processed or blacklisted videos
blacklist_set = set()
processed_set = set()

# Read existing entries from BLACKLIST.txt and PROCESSED.txt
if os.path.exists("./BLACKLIST.txt"):
    with open("./BLACKLIST.txt", "r", encoding="utf-8") as f:
        blacklist_set = set(line.strip() for line in f if line.strip())

if os.path.exists("./PROCESSED.txt"):
    with open("./PROCESSED.txt", "r", encoding="utf-8") as f:
        processed_set = set(line.strip() for line in f if line.strip())

# Collect video files, excluding those in blacklist or processed sets
VIDEO_FILES = []
for root, _, files in os.walk(INPUT_VIDEO_PATH):
    for file in files:
        if file.endswith(".mp4"):
            full_path = os.path.join(root, file)
            if (
                os.path.exists(full_path)
                and full_path not in blacklist_set
                and full_path not in processed_set
            ):
                VIDEO_FILES.append(full_path)

# Face detection and video codec
FACE_DETECTOR = dlib.get_frontal_face_detector()
CODEC = cv2.VideoWriter_fourcc(*"mp4v")


# Function to process a single video
def process_video(in_video):
    try:
        cap = cv2.VideoCapture(in_video)
        if not cap.isOpened():
            print(f"Failed to open video: {in_video}")
            return ("blacklist", in_video)

        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Build output file path
        output_file_path = os.path.join(
            OUTPUT_VIDEO_PATH, os.path.relpath(in_video, INPUT_VIDEO_PATH)
        )

        # Ensure output directory exists
        output_dir = os.path.dirname(output_file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"Processing video: {in_video}, Output path: {output_file_path}")

        out_video = cv2.VideoWriter(
            output_file_path, CODEC, fps, (frame_width, frame_height)
        )

        if not out_video.isOpened():
            print(f"Failed to write video: {output_file_path}")
            return ("blacklist", in_video)

        # Read and process each frame
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                # print(f"End of video file or read error at frame {frame_count}")
                break

            frame_count += 1

            if len(FACE_DETECTOR(frame)) == 0:  # Check for no faces detected
                # print(f"No faces detected in frame {frame_count}")
                continue

            out_video.write(frame)

        # Clean up
        out_video.release()
        cap.release()
        print(f"Finished processing video: {in_video}")
        return ("processed", in_video)

    except Exception as e:
        print(f"Error processing video {in_video}: {e}")
        return ("blacklist", in_video)


# Use ProcessPoolExecutor to process videos in parallel
if __name__ == "__main__":
    results = []
    with ProcessPoolExecutor() as executor:
        # Map the process_video function to the list of video files
        for result in executor.map(process_video, VIDEO_FILES):
            results.append(result)

    # Write the results to BLACKLIST.txt and PROCESSED.txt
    with open("./BLACKLIST.txt", "a", encoding="utf-8") as blacklist, open(
        "./PROCESSED.txt", "a", encoding="utf-8"
    ) as processed:
        for result_type, video_path in results:
            if result_type == "blacklist":
                blacklist.write(video_path + "\n")
            elif result_type == "processed":
                processed.write(video_path + "\n")

    print("Processing complete.")
