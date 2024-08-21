import cv2
import transformers
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

input_video_path = r"G:\My Drive\Megatron_DeepFake\dataset\original_sequences\actors\raw\videos\01__kitchen_pan.mp4"
output_video_path = "./data/dataset/depth_01__kitchen_pan.mp4"
pipeline = transformers.pipeline(
    task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf"
)
# Open the input video
capture_card = cv2.VideoCapture(input_video_path)


# Check if the video capture and writer are opened successfully
if not capture_card.isOpened():
    print("Error: Could not open video file.")
cnt = 1
# Process each frame
while capture_card.isOpened():

    ret, frame = capture_card.read()
    if not ret or cnt % 10 == 0:
        break

    frame = Image.fromarray(frame)

    depth = pipeline(frame)["depth"]
    depth = np.array(depth)
    plt.imshow(depth)
    plt.show()
    # Write the processed frame to the output video
    cnt += 1

# Release resources
capture_card.release()
