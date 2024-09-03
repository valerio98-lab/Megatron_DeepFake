"""Module containing the definition for the dasaet and 
dataloader"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Optional, Union
import random

import cv2
import dlib  # type: ignore
from transformers import Pipeline  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from megatron.preprocessing import RepVit, PositionalEncoding
from megatron.transformations import TRANSFORMATIONS


@dataclass
class Video:
    """
    Represents a video with RGB and depth frames.

    Attributes:
        rgb_frames (torch.Tensor): A tensor containing RGB frames of the video.
        depth_frames (torch.Tensor): A tensor containing depth frames of the video.
        original (bool): Indicates whether the video is an original or not.
    """

    rgb_frames: torch.Tensor
    depth_frames: torch.Tensor
    original: bool


class VideoDataset(Dataset):
    """
    A PyTorch dataset for loading video data.

    Args:
        video_dir (Path): The directory path where the video files are located.
        depth_anything_size (Literal["Small", "Base", "Large"], optional): The size of the depth-anything model to use.
            Defaults to "Small".
        num_video (int | None, optional): The maximum number of videos to load.
            If None, all videos in the directory will be loaded. Defaults to None.
        num_frame (int, optional): The number of frames to extract from each video.
            Defaults to 1.
        random_initial_frame (bool, optional): Whether to randomly select the initial frame for extraction.
            Defaults to False.
    """

    def __init__(
        self,
        video_dir: Path,
        depth_anything: Pipeline,
        num_video: int | None = None,
        num_frame: int = 1,
        random_initial_frame: bool = False,
        techniques: Optional[list[str]] = None,
    ):
        super().__init__()
        self.num_frame = num_frame
        self.video_dir = Path(video_dir)
        self.num_video = num_video
        self.random_initial_frame = random_initial_frame
        self.techniques = techniques
        self.video_paths = self.__collate_video()
        self.face_detector = dlib.get_frontal_face_detector()  # type: ignore
        self.depth_anything = depth_anything

    def __len__(self) -> int:
        return len(self.video_paths)

    def __collate_video(self) -> list[str]:
        original_video_paths = []
        manipulated_video_paths = []
        if str(self.video_dir).endswith(".mp4"):
            return [str(self.video_dir)]

        for root, _, files in os.walk(self.video_dir):
            if self.techniques is not None and not any(
                pair[0] in pair[1]
                for pair in zip(self.techniques, [root] * len(self.techniques))
            ):
                continue
            for file in files:
                if file.endswith(".mp4"):
                    video_path = os.path.join(root, file)

                    if "original" in video_path:
                        original_video_paths.append(video_path)
                    elif "manipulated" in video_path:
                        manipulated_video_paths.append(video_path)

        video_paths = (original_video_paths + manipulated_video_paths) * len(
            TRANSFORMATIONS
        )
        np.random.shuffle(video_paths)
        if self.num_video is not None and (self.num_video * len(TRANSFORMATIONS)) <= (
            len(video_paths)
        ):
            indxs = torch.randperm(self.num_video * len(TRANSFORMATIONS))
        else:
            indxs = indxs = torch.randperm(len(video_paths))
        return np.array(video_paths)[indxs].tolist()

    def __getitem__(self, idx: int) -> Union[Video, None]:
        try:
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
        # Bad practice, but we'll manage cases as they appear
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error loading video: {video_path}, {e}")
            return None

    def extract_frames_and_faces(
        self, cap: cv2.VideoCapture, length: int
    ) -> tuple[list[torch.Tensor], list[np.ndarray]]:
        """
        Extracts frames and faces from a video.

        Args:
            cap (cv2.VideoCapture): The video capture object.
            length (int): The number consecutive of frames to extract.

        Returns:
            tuple[list[torch.Tensor], list[np.ndarray]]: A tuple containing a list of RGB
                frames as torch Tensors and a list of face crops as numpy arrays.
        """
        transformation, gen_kwargs = random.choice(TRANSFORMATIONS)
        kwargs = gen_kwargs()
        rgb_frames, face_crops = [], []
        for _ in range(length):
            ret, frame = cap.read()
            if not ret:
                break
            frame = transformation(frame, **kwargs)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_crop = self.face_extraction(frame_rgb)
            if face_crop is None:
                break
            face_crops.append(face_crop)
            rgb_frames.append(torch.from_numpy(face_crop).permute(2, 0, 1))
        return rgb_frames, face_crops

    def face_extraction(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts the face from the given frame.

        Parameters:
            frame (np.ndarray): The input frame containing the face.

        Returns:
            np.ndarray | None: The extracted face as a numpy array, or None if no face is found.
        """
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
        """
        Calculate depth frames for a list of face crops.
        Args:
            face_crops (list[np.ndarray]): A list of numpy arrays representing face crops.
        Returns:
            list[torch.Tensor]: A list of torch Tensors representing the depth frames.
        """
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
        """
        Pads the given RGB and depth frames to the maximum dimensions.

        Args:
            rgb_frames (list[torch.Tensor]): A list of RGB frames.
            depth_frames (list[torch.Tensor]): A list of depth frames.

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: A tuple containing the
                padded RGB frames and padded depth frames.
        """
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
        """
        Pads the frames in the list to match the maximum height and width.

        Args:
            frames (list[torch.Tensor]): A list of torch.Tensor representing frames.
            max_height (int): The maximum height to pad the frames to.
            max_width (int): The maximum width to pad the frames to.

        Returns:
            list[torch.Tensor]: A list of torch.Tensor with padded frames.

        """
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
    """
    A custom data loader for loading video data.

    Attributes:
        repvit (RepVit): The RepVit model for processing video frames.
        positional_encoding (PositionalEncoding): The positional encoder for encoding video frames.
        device (torch.device): The device (CPU or GPU) to use for processing.
    """

    def __init__(
        self,
        dataset: Union[VideoDataset, Subset],
        repvit: RepVit,
        positional_encoding: PositionalEncoding,
        batch_size: int = 1,
        shuffle: bool = True,
        pin_memory: bool = True,
        num_workers: int = 4,
    ):
        """Initializes a VideoDataLoader instance.
        Args:
            dataset (VideoDataset): The video dataset to load.
            repvit (RepVit): The RepVit model for processing video frames.
            positional_encoding (PositionalEncoding): The positional encoder for encoding video frames.
            batch_size (int, optional): The batch size for loading data. Defaults to 1.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            pin_memory (bool, optional): Whether to pin memory for faster data transfer. Defaults to True.
            num_workers (int, optional): The number of worker processes for data loading. Defaults to 4.
        """
        self.repvit = repvit
        self.positional_encoding = positional_encoding
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.shuffle = shuffle
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.__collate_fn,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )

    def __collate_fn(
        self, batch: list[Video]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
                video_depth_frames = self.positional_encoding(video_depth_frames)
                depth_frames.append(video_depth_frames)

                # Processing RGB frames
                video_rgb_frames = video.rgb_frames.to(self.device)
                video_rgb_frames = self.repvit(video_rgb_frames)
                video_rgb_frames = self.positional_encoding(video_rgb_frames)
                rgb_frames.append(video_rgb_frames)

                labels.append(int(video.original))
        depth_frames_tensor = torch.stack(depth_frames)
        rgb_frames_tensor = torch.stack(rgb_frames)
        labels_tensor = torch.tensor(labels, device=self.device)
        return rgb_frames_tensor, depth_frames_tensor, labels_tensor

    # The only purpose for this is for helping pylint with type annotations.
    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:  # type: ignore
        return super().__iter__()

    # Implement slicing
    def __getitem__(self, index):
        if isinstance(index, slice):
            # Handle slicing
            start, stop, step = index.indices(len(self.dataset))
            indices = range(start, stop, step)
        else:
            # Handle single index
            indices = [index]

        # Create a Subset of the dataset based on indices
        subset = Subset(self.dataset, indices)

        # Create a new DataLoader for the subset
        return VideoDataLoader(
            dataset=subset,
            repvit=self.repvit,
            positional_encoding=self.positional_encoding,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )


# VideoDataset(
#         video_dir=Path(
#             r"G:\\My Drive\\Megatron_DeepFake\\dataset\\original_sequences\\youtube\\raw\\videos\\456.mp4"
#         ),
#         depth_anything=None,
#         num_frame=30,
#         num_video=100,
#         random_initial_frame=False,
#     )[0]
