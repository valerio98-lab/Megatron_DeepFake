import os
import pathlib
import cv2
import dlib
import torch
import transformers
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Literal


class VideoDataset(Dataset):
    def __init__(
        self,
        video_dir: os.PathLike,
        depth_anything_size: Literal["Small", "Base", "Large"] = "Small",
        num_video: int | None = None,
        threshold: int = 5,
        num_frame: int = 1,
    ):
        super().__init__()
        self.num_frame = num_frame
        self.threshold = threshold
        self.data_path = pathlib.Path(video_dir)
        self.num_video = num_video

        assert (
            self.data_path.exists()
        ), f'Watch out! "{str(self.data_path)}" was not found.'

        self.video_files = []
        self.face_detector = dlib.get_frontal_face_detector()
        self.pipeline = transformers.pipeline(
            task="depth-estimation",
            model=f"depth-anything/Depth-Anything-V2-{depth_anything_size}-hf",
        )

        self.video_files = self.__collate_video()

    def __len__(self):
        return len(self.video_files)

    def __collate_video(self):
        cnt = 0
        video_files = []
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if cnt >= self.num_video and self.num_video is not None:
                    return video_files
                if file.endswith(".mp4"):
                    cnt += 1
                    complete_path = os.path.join(root, file)
                    label = "manipulated" in complete_path
                    video_files.append((complete_path, label))
        return video_files

    def __getitem__(self, idx):
        video_path, label = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)
        # get the number of frames in the video and set the length to the minimum between the number of frames and the number of frames we want to extract.
        length = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), self.num_frame)
        frames = []
        for _ in range(length):
            if cap.isOpened():
                ret, frame = (
                    cap.read()
                )  # ret is a boolean value that indicates if the frame was read correctly or not. frame is the image in BGR format.
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # face cropping operations
                face_crop = self.face_extraction(frame)
                if face_crop is None:
                    break
                # depth map operations on face_crop
                depth_mask = self.calculate_depth_mask(face_crop)
                # convert to tensor for RepVit model
                face_crop = face_crop.clone().detach()
                depth_mask = depth_mask.clone().detach()

                frames.append((face_crop, depth_mask, label))

        cap.release()
        if len(frames) >= self.threshold:
            return frames

    def face_extraction(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
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


class VideoDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        RepVit_model=None,
        batch_size=1,
        shuffle=True,
        custom_collate_fn=None,
    ):
        self.dataset = dataset
        self.RepVit = RepVit_model
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = (
            self.__collate_fn if custom_collate_fn is None else custom_collate_fn
        )
        super().__init__(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
        )

    def __collate_fn(self, batch):
        embedded_batch = []
        for video in batch:
            embedded_frames = []
            for rgb_crop, depth_mask, label in video:
                rgb_crop = rgb_crop.unsqueeze(0)
                depth_mask = depth_mask.unsqueeze(0)

                with torch.no_grad():
                    embedded_rgb = self.get_repvit_embedding(rgb_crop)
                    embedded_depth = self.get_repvit_embedding(depth_mask)

                embedded_frames.append((embedded_rgb, embedded_depth, label))
            embedded_batch.append(embedded_frames)

        return embedded_batch

    def get_repvit_embedding(self, tensor):
        # return self.RepVit(tensor)
        return torch.rand(1, 10, 10)

    def DataLoader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
        )


if __name__ == "__main__":
    VIDEO_PATH = r"G:\My Drive\Megatron_DeepFake\dataset"
    DEPTH_ANYTHING_SIZE = "Small"
    NUM_FRAMES = 5
    BATCH_SIZE = 2
    SHUFFLE = True
    dataset = VideoDataset(
        VIDEO_PATH, DEPTH_ANYTHING_SIZE, num_frame=NUM_FRAMES, num_video=BATCH_SIZE
    )
    dataloader = VideoDataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    for batch in dataloader:
        print(len(batch))
        for elem in batch:
            for x in elem:
                print(len(x), end=" ")
                fig, axs = plt.subplots(2, 2, figsize=(10, 10))
                plt.title("original" if x[2] else "manipulated")
                print(type(x[0]), type(x[1]), type(x[2]))
