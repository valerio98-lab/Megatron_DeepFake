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
        cnt_original = 0
        video_paths = []
        if str(self.data_path).endswith(".mp4"):
            return [str(self.data_path)]
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.endswith(".mp4"):
                    cnt_original += 1 if "original" in file else 0
                    video_path = os.path.join(root, file)
                    video_paths.append(video_path)
        cnt_manipulated = len(video_paths) - cnt_original + 1
        cnt_original /= len(video_paths)

        cnt_manipulated /= len(video_paths)
        # distribution = torch.distributions.Categorical(torch.tensor([cnt_original, cnt_manipulated]))
        print(f"{self.num_video=}")
        if self.num_video is not None and self.num_video <= len(video_paths):
            indxs = torch.randperm(self.num_video)
            return np.array(video_paths)[
                indxs
            ].tolist()  # video_paths[: self.num_video]
        indxs = torch.randperm(len(video_paths))
        return np.array(video_paths)[indxs].tolist()

    def __getitem__(self, idx: int) -> Union[Video, None]:
        video_path = self.video_paths[idx]
        label = self.get_label(video_path)
        cap = self.open_video_capture(video_path)
        if not cap:
            return None

        total_frame, length = self.get_video_length(cap)
        if self.random_initial_frame:
            self.set_random_start_frame(cap, total_frame, length)

        rgb_frames, face_crops = self.extract_frames_and_faces(cap, length)
        cap.release()

        # if len(rgb_frames) < self.threshold:
        if len(rgb_frames) < self.num_frame:

            return None

        depth_frames = self.calculate_depth_frames(face_crops)
        rgb_frames, depth_frames = self.pad_frames(rgb_frames, depth_frames)

        return Video(
            rgb_frames=torch.stack(rgb_frames),
            depth_frames=torch.stack(depth_frames),
            original=label,
        )

    def get_label(self, video_path: str) -> bool:
        return "original" in video_path

    def open_video_capture(self, video_path: str) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(video_path)
        return cap if cap.isOpened() else None

    def get_video_length(self, cap: cv2.VideoCapture) -> tuple[int, int]:
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        length = min(total_frame, self.num_frame)
        return total_frame, length

    def set_random_start_frame(
        self, cap: cv2.VideoCapture, total_frame: int, length: int
    ) -> None:
        random_frame = int(np.random.uniform(0, total_frame - length))
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)

    def extract_frames_and_faces(
        self, cap: cv2.VideoCapture, length: int
    ) -> tuple[list[torch.Tensor], list[np.ndarray]]:
        rgb_frames, face_crops = [], []
        for _ in range(length):
            ret, frame = cap.read()
            if not ret:
                break
            face_crop = self.process_frame(frame)
            if face_crop is None:
                break
            face_crops.append(face_crop)
            rgb_frames.append(self.convert_to_tensor(face_crop))
        return rgb_frames, face_crops

    def process_frame(self, frame: np.ndarray) -> Union[np.ndarray, None]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.face_extraction(frame_rgb)

    def convert_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(image).permute(2, 0, 1)  # Change to CxHxW

    def calculate_depth_frames(
        self, face_crops: list[np.ndarray]
    ) -> list[torch.Tensor]:
        depth_masks = self.calculate_depth_masks(face_crops)
        return [self.convert_to_tensor(depth_mask) for depth_mask in depth_masks]

    def pad_frames(
        self, rgb_frames: list[torch.Tensor], depth_frames: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        max_height, max_width = self.get_max_dimensions(rgb_frames)
        rgb_frames_padded = self.pad_to_max_dimensions(
            rgb_frames, max_height, max_width
        )
        depth_frames_padded = self.pad_to_max_dimensions(
            depth_frames, max_height, max_width
        )
        return rgb_frames_padded, depth_frames_padded

    def get_max_dimensions(self, frames: list[torch.Tensor]) -> tuple[int, int]:
        max_height = max(frame.size(1) for frame in frames)
        max_width = max(frame.size(2) for frame in frames)
        return max_height, max_width

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

    def face_extraction(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Given a frame, if a face is found it returns the image cropped around the face,
        or else returns None.

        Args:
            frame (np.ndarray): The input frame containing the image.

        Returns:
            (np.ndarray | None): If a face is found, it returns the cropped image around the face.
                If no face is found, it returns None.
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

    def calculate_depth_masks(self, faces: list[np.ndarray]) -> list[np.ndarray]:
        """
        Calculates the depth masks for a list of face images.

        Args:
            faces (List[numpy.ndarray]): A list of face images as NumPy arrays.

        Returns:
            List[numpy.ndarray]: A list of depth masks as NumPy arrays.
        """
        # Convert the list of NumPy arrays to a list of PIL images
        pil_faces = [Image.fromarray(face) for face in faces]

        # Run the depth estimation pipeline on all faces at once
        results = self.depth_anything(pil_faces)

        # Extract depth masks and convert them back to NumPy arrays
        depth_masks = [
            np.stack((np.array(result["depth"]),) * 3, axis=-1) for result in results
        ]

        return depth_masks


class VideoDataLoader(DataLoader):
    """
    A custom data loader for loading video datasets.

    Args:
        dataset (VideoDataset): The video dataset to load.
        repvit_model (Literal[str]): The name of the RepVit model to use for embedding extraction.
            Default is "repvit_m0_9.dist_300e_in1k".
        batch_size (int): The batch size for loading the data. Default is 1.
        shuffle (bool): Whether to shuffle the data. Default is True.
        custom_collate_fn (callable): A custom collate function to use for batching the data.
            If None, the default collate function will be used. Default is None.

    """

    def __init__(
        self,
        dataset: VideoDataset,
        repvit,
        positional_encoder,
        batch_size=1,
        shuffle=True,
        custom_collate_fn=None,
    ):
        self.dataset = dataset
        self.repvit = repvit
        self.positional_encoder = positional_encoder
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

    def __collate_fn(self, batch: list[Video]):
        batch = list(filter(None, batch))
        return batch

    def __len__(self) -> int:
        return len(self.dataset)

    # The only purpose for this is for helping pylint with type annotations.
    def __iter__(self) -> Iterator[list[Video]]:
        return super().__iter__()


if __name__ == "__main__":
    from megatron.trainer import Config, Trainer
    import torch

    experiment = {
        "dataset": {
            "video_path": r"H:\My Drive\Megatron_DeepFake\dataset",
            "num_frames": 5,
            "random_initial_frame": True,
            "depth_anything_size": "Small",
            "num_video": 40,
            "frame_threshold": 10,
        },
        "dataloader": {
            "batch_size": 4,
            "repvit_model": "repvit_m0_9.dist_300e_in1k",
        },
        "transformer": {
            "d_model": 384,
            "n_heads": 2,
            "n_layers": 1,
            "d_ff": 1024,
        },
        "train": {
            "learning_rate": 0.001,
            "epochs": 2,
            "log_dir": "../data/runs/exp1",
            "early_stop_counter": 10,
            "train_size": 0.5,
            "val_size": 0.3,
            "test_size": 0.2,
            "seed": 42,
        },
    }

    config = Config(**experiment)
    torch.manual_seed(config.train.seed)
    trainer = Trainer(config)
    trainer.train()
