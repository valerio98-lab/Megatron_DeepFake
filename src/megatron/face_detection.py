import cv2
import dlib

# Initialize the face detector
detector = dlib.get_frontal_face_detector()

input_video_path = r"G:\My Drive\Megatron_DeepFake\dataset\original_sequences\actors\raw\videos\01__kitchen_pan.mp4"
output_video_path = "./data/dataset/01__kitchen_pan.mp4"

# Open the input video
capture_card = cv2.VideoCapture(input_video_path)

# Get the video's width, height, and frames per second (fps)
frame_width = int(capture_card.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture_card.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = capture_card.get(cv2.CAP_PROP_FPS)

# Initialize the video writer to save the output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for mp4
output_video = cv2.VideoWriter(
    output_video_path, fourcc, fps, (frame_width, frame_height)
)

# Check if the video capture and writer are opened successfully
if not capture_card.isOpened():
    print("Error: Could not open video file.")
if not output_video.isOpened():
    print("Error: Could not open video writer.")

# Process each frame
while capture_card.isOpened():
    ret, frame = capture_card.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(frame_gray)

    # Draw bounding boxes around each detected face
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Write the processed frame to the output video
    output_video.write(frame)

# Release resources
capture_card.release()
output_video.release()
