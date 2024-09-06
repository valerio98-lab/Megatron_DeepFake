# https://www.blog.pythonlibrary.org/2021/06/29/converting-mp4-to-animated-gifs-with-python/
import cv2
from PIL import Image
import dlib


def convert_mp4_to_jpgs(path):
    images = []
    video_capture = cv2.VideoCapture(path)
    if not video_capture.isOpened():
        raise ValueError("Error opening")
    still_reading, image = video_capture.read()
    frame_count = 0
    print(f"Converting {path}")
    while still_reading and frame_count <= (50):
        # read next image
        still_reading, image = video_capture.read()
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frame_count += 1
    video_capture.release()
    return images


def convert_mp4_to_jpgs_with_faces(path):
    # Initialize the dlib face detector
    detector = dlib.get_frontal_face_detector()

    images = []
    video_capture = cv2.VideoCapture(path)
    if not video_capture.isOpened():
        raise ValueError("Error opening video")

    still_reading, image = video_capture.read()
    frame_count = 0
    print(f"Converting {path}")
    while still_reading and frame_count <= 50:
        # Read next image
        still_reading, image = video_capture.read()
        if not still_reading:
            break

        # Convert color from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = detector(image_rgb, 1)

        # Draw a green bounding box around each detected face
        for face in faces:
            cv2.rectangle(
                image_rgb,
                (face.left(), face.top()),
                (face.right(), face.bottom()),
                (0, 255, 0),
                2,
            )

        images.append(image_rgb)
        frame_count += 1

    video_capture.release()
    return images


def make_gif(images, output):
    images = [Image.fromarray(frame) for frame in images]
    frame_one = images[0]
    frame_one.save(
        output,
        format="GIF",
        append_images=images,
        save_all=True,
        duration=50,
        loop=1,
    )


if __name__ == "__main__":
    # #     frames = convert_mp4_to_jpgs(
    # #         r"H:\My Drive\Megatron_DeepFake\dataset_processed\original_sequences\youtube\raw\videos\000.mp4"
    # #     )
    # #     make_gif(frames, "./assets/original_sample.gif")

    # #     frames = convert_mp4_to_jpgs(
    # #         r"H:\My Drive\Megatron_DeepFake\dataset_processed\manipulated_sequences\Deepfakes\raw\videos\000_003.mp4"
    # #     )
    # #     make_gif(frames, "./assets/deepfake_sample.gif")

    # #     frames = convert_mp4_to_jpgs(
    # #         r"H:\My Drive\Megatron_DeepFake\dataset_processed\manipulated_sequences\Face2Face\raw\videos\000_003.mp4"
    # #     )
    # #     make_gif(frames, "./assets/face2face_sample.gif")

    frames = convert_mp4_to_jpgs_with_faces(
        r"H:\My Drive\Megatron_DeepFake\dataset_processed\original_sequences\youtube\raw\videos\448.mp4"
    )
    make_gif(frames, "./assets/test_sample_dlib.gif")
