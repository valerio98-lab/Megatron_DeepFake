import os
import cv2
from pathlib import Path
import dlib
import torch
import transformers
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from enum import StrEnum


class DepthAnythingSize(StrEnum):
    SMALL = "Small"
    BASE = "Base"
    LARGE = "Large"


class VideoDataset(Dataset):
    def __init__(self, video_path, depth_anything_size, num_frame=1):
        super().__init__()
        self.num_frame = num_frame
        self.video_path = Path(video_path)
        assert (
            self.video_path.exists()
        ), f'Watch out! "{str(self.video_path)}" was not found.'

        self.videos = []
        for root, _, files in os.walk("video_path"):
            for file in files:
                if file.endswith(".mp4"):
                    self.videos.append(os.path.join(root, file))

        self.face_detector = dlib.get_frontal_face_detector()
        self.pipeline = transformers.pipeline(
            task="depth-estimation",
            model=f"depth-anything/Depth-Anything-V2-{depth_anything_size}-hf",
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        cap = cv2.VideoCapture(str(video_path))
        faces = []
        face_crop = None
        depth_mask = None
        while len(faces) < self.num_frame:
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = self.cap.read()
                # ret is a boolean value that indicates if the frame was read correctly or not. frame is the image in BGR format.
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # face cropping operations
                    face_crop = self.face_extraction(frame)
                    if face_crop is None and len(faces) != 0:
                        break
                    elif face_crop is None and len(faces) > 0:
                        continue
                    # depth map operations on face_crop
                    depth_mask = self.calculate_depth_mask(face_crop)
                faces.append((face_crop, depth_mask))

        return faces, "original" in video_path

    def face_extraction(self, frame):
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector(frame)
        face = None
        for face_rect in faces:
            x, y, w, h = (
                face_rect.left(),
                face_rect.top(),
                face_rect.width(),
                face_rect.height(),
            )
            face = frame[y : y + h, x : x + w]
            break  # We decided to keep only one face in case more where present.
        return face

    def calculate_depth_mask(self, face):
        face = Image.fromarray(face)
        depth = self.pipeline(face)["depth"]
        depth = np.array(depth)
        return depth


if __name__ == "__main__":
    video_path = r"G:\My Drive\Megatron_DeepFake\dataset\original_sequences\youtube\raw\videos\159.mp4"
    video_dataset = VideoDataset(video_path)
    face1, depth1 = video_dataset[0]

    video_path = r"G:\My Drive\Megatron_DeepFake\dataset\manipulated_sequences\FaceSwap\raw\videos\159_175.mp4"
    video_dataset = VideoDataset(video_path)
    face2, depth2 = video_dataset[0]
    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Display the images in the subplots
    axs[0, 0].imshow(face1)
    axs[0, 0].axis("off")  # Hide the axis

    axs[0, 1].imshow(face2)
    axs[0, 1].axis("off")

    axs[1, 0].imshow(depth1)
    axs[1, 0].axis("off")

    axs[1, 1].imshow(depth2)
    axs[1, 1].axis("off")

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
