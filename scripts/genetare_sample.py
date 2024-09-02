# Disable all pylint checks
# pylint: disable=all
import cv2

# Path to the input video
input_video_path = r"G:\\My Drive\\Megatron_DeepFake\\dataset\\original_sequences\\youtube\\raw\\videos\\456.mp4"

# Path to the output video
output_video_path = "./assets/sample.mp4"

# Open the video
cap = cv2.VideoCapture(input_video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video frames per second (fps) and size for the output video
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Frame index counter
frame_idx = 0

# Read frames from the video
while cap.isOpened():
    ret, frame = cap.read()
    print(ret)
    if not ret:
        break

    # Check if the current frame index is within the desired range
    if 10 <= frame_idx <= 29:
        out.write(frame)  # Write frame to the output video

    frame_idx += 1

    # Break the loop if we have processed frame 29
    if frame_idx > 29:
        break

# Release everything
cap.release()
out.release()

print("Video processing complete. Output saved to:", output_video_path)
