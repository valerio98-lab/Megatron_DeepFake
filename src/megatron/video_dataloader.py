"""Module containing the definition for the dasaet and 
dataloader"""

import os
import pathlib
from dataclasses import dataclass
from typing import Iterator, Union

import cv2
import dlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


@dataclass
class Video:
    rgb_frames: torch.Tensor
    depth_frames: torch.Tensor
    original: bool


class VideoDataset(Dataset):
    """
    A PyTorch dataset for loading video data.

    Args:
        video_dir (os.PathLike): The directory path where the video files are located.
        depth_anything_size (Literal["Small", "Base", "Large"], optional): The size of the depth-anything model to use.
            Defaults to "Small".
        num_video (int | None, optional): The maximum number of videos to load.
            If None, all videos in the directory will be loaded. Defaults to None.
        threshold (int, optional): The minimum number of frames required for a video to be considered valid.
            Defaults to 5.
        num_frame (int, optional): The number of frames to extract from each video.
            Defaults to 1.
        random_initial_frame (bool, optional): Whether to randomly select the initial frame for extraction.
            Defaults to False.
    """

    def __init__(
        self,
        video_dir: os.PathLike,
        depth_anything,
        num_video: int | None = None,
        threshold: int = 1,
        num_frame: int = 1,
        random_initial_frame: bool = False,
    ):
        super().__init__()
        self.num_frame = num_frame
        self.threshold = threshold
        self.data_path = pathlib.Path(video_dir)
        self.num_video = num_video
        self.random_initial_frame = random_initial_frame
        assert (
            self.data_path.exists()
        ), f'Watch out! "{str(self.data_path)}" was not found.'

        self.video_paths = []
        self.video_paths = self.__collate_video()
        self.face_detector = dlib.get_frontal_face_detector()
        self.depth_anything = depth_anything

    def __len__(self) -> int:
        return len(self.video_paths)

    def __collate_video(self) -> list[str]:
        original_video_paths = []
        manipulated_video_paths = []

        if str(self.data_path).endswith(".mp4"):
            return [str(self.data_path)]

        # TODO: Jose,Valerio, trovare un modo piu intelligente
        # per la randomizzazione del dataset
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.endswith(".mp4"):
                    video_path = os.path.join(root, file)

                    if "original" in video_path:
                        original_video_paths.append(video_path)
                    elif "manipulated" in video_path:
                        manipulated_video_paths.append(video_path)
        video_paths = original_video_paths + manipulated_video_paths
        np.random.shuffle(video_paths)

        if self.num_video is not None and self.num_video <= (
            len(original_video_paths) + len(manipulated_video_paths)
        ):
            indxs = torch.randperm(self.num_video)
        else:
            indxs = indxs = torch.randperm(len(video_paths))
        return np.array(video_paths)[indxs].tolist()

    def __getitem__(self, idx: int) -> Union[Video, None]:
        video_path = self.video_paths[idx]
        label = "original" in video_path
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        length = min(total_frame, self.num_frame)
        if length < self.num_frame:
            return None

        if self.random_initial_frame:
            random_frame = int(np.random.uniform(0, total_frame - length))
            cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)

        rgb_frames, face_crops = self.extract_frames_and_faces(cap, length)
        cap.release()

        if len(rgb_frames) < self.num_frame:
            return None

        depth_frames = self.calculate_depth_frames(face_crops)
        rgb_frames, depth_frames = self.pad_frames(rgb_frames, depth_frames)

        return Video(
            rgb_frames=torch.stack(rgb_frames),
            depth_frames=torch.stack(depth_frames),
            original=label,
        )

    def extract_frames_and_faces(
        self, cap: cv2.VideoCapture, length: int
    ) -> tuple[list[torch.Tensor], list[np.ndarray]]:
        rgb_frames, face_crops = [], []
        for _ in range(length):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_crop = self.face_extraction(frame_rgb)
            if face_crop is None:
                break
            face_crops.append(face_crop)
            rgb_frames.append(torch.from_numpy(face_crop).permute(2, 0, 1))
        return rgb_frames, face_crops

    def face_extraction(self, frame: np.ndarray) -> np.ndarray | None:
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

    def calculate_depth_frames(
        self, face_crops: list[np.ndarray]
    ) -> list[torch.Tensor]:
        pil_faces = [Image.fromarray(face) for face in face_crops]

        # Run the depth estimation pipeline on all faces at once
        results = self.depth_anything(pil_faces)

        # Extract depth masks and convert them back to NumPy arrays
        depth_masks = [
            np.stack((np.array(result["depth"]),) * 3, axis=-1) for result in results
        ]
        return [
            torch.from_numpy(depth_mask).permute(2, 0, 1) for depth_mask in depth_masks
        ]

    def pad_frames(
        self, rgb_frames: list[torch.Tensor], depth_frames: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        max_height = max(frame.size(1) for frame in rgb_frames)
        max_width = max(frame.size(2) for frame in rgb_frames)
        rgb_frames_padded = self.pad_to_max_dimensions(
            rgb_frames, max_height, max_width
        )
        depth_frames_padded = self.pad_to_max_dimensions(
            depth_frames, max_height, max_width
        )
        return rgb_frames_padded, depth_frames_padded

    def pad_to_max_dimensions(
        self, frames: list[torch.Tensor], max_height: int, max_width: int
    ) -> list[torch.Tensor]:
        return [
            (
                F.pad(
                    frame,
                    (0, max_width - frame.size(2), 0, max_height - frame.size(1)),
                    mode="constant",
                    value=0,
                )
                if frame.size(1) != max_height or frame.size(2) != max_width
                else frame
            )
            for frame in frames
        ]


class VideoDataLoader(DataLoader):

    def __init__(
        self,
        dataset: VideoDataset,
        repvit,
        positional_encoder,
        batch_size=1,
        shuffle=True,
    ):
        self.repvit = repvit
        self.positional_encoder = positional_encoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.__collate_fn,
        )

    def __collate_fn(self, batch: list[Video]):
        labels = []
        depth_frames = []
        rgb_frames = []
        with torch.no_grad():
            for video in batch:
                if video is None:
                    continue

                # Processing depth frames
                video_depth_frames = video.depth_frames.to(self.device)
                video_depth_frames = self.repvit(video_depth_frames)
                video_depth_frames = self.positional_encoder(video_depth_frames)
                depth_frames.append(video_depth_frames)

                # Processing RGB frames
                video_rgb_frames = video.rgb_frames.to(self.device)
                video_rgb_frames = self.repvit(video_rgb_frames)
                video_rgb_frames = self.positional_encoder(video_rgb_frames)
                rgb_frames.append(video_rgb_frames)

                labels.append(int(video.original))
        depth_frames = torch.stack(depth_frames)
        rgb_frames = torch.stack(rgb_frames)
        labels = torch.tensor(labels, device=self.device)
        return rgb_frames, depth_frames, labels

    def __len__(self) -> int:
        return len(self.dataset)

    # The only purpose for this is for helping pylint with type annotations.
    def __iter__(self) -> Iterator[list[Video]]:
        return super().__iter__()
