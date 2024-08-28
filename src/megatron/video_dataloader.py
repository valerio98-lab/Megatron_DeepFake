"""Module containing the definition for the dasaet and 
dataloader"""

import os
import pathlib
from typing import Iterator, Literal, Union
from dataclasses import dataclass
import torch.nn.functional as F

import cv2
import dlib
import numpy as np
import torch
import transformers
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
        depth_anything_size: Literal["Small", "Base", "Large"] = "Small",
        num_video: int | None = None,
        threshold: int = 5,
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
        self.pipeline = transformers.pipeline(
            task="depth-estimation",
            model=f"depth-anything/Depth-Anything-V2-{depth_anything_size}-hf",
        )

    def __len__(self) -> int:
        return len(self.video_paths)

    def __collate_video(self) -> list[(str, bool)]:
        cnt = 0
        video_paths = []
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if self.num_video is not None and cnt >= self.num_video:
                    return video_paths
                if file.endswith(".mp4"):
                    cnt += 1
                    video_path = os.path.join(root, file)
                    video_paths.append(video_path)
        return video_paths

    def __getitem__(self, idx: int) -> Union[Video, None]:
        video_path = self.video_paths[idx]
        label = "manipulated" in video_path
        cap = cv2.VideoCapture(video_path)
        # Get the number of frames in the video and set the length to the minimum between
        # the number of frames and the number of frames we want to extract.
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        length = min(total_frame, self.num_frame)
        if self.random_initial_frame:
            cap.set(
                cv2.CAP_PROP_POS_FRAMES, int(np.random.uniform(0, total_frame - length))
            )

        rgb_frames = []
        depth_frames = []
        if cap.isOpened():
            for _ in range(length):
                # ret is a boolean value that indicates if the frame was read correctly or not.
                # frame is the image in BGR format.
                ret, frame = cap.read()

                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Face cropping operations
                face_crop = self.face_extraction(frame)

                if face_crop is None:
                    break
                # Depth map operations on face_crop
                depth_mask = self.calculate_depth_mask(face_crop)
                # Convert to tensor for RepVit model
                face_crop = torch.from_numpy(face_crop)
                depth_mask = torch.from_numpy(depth_mask)

                rgb_frames.append(face_crop)
                depth_frames.append(depth_mask)

            cap.release()

            if len(rgb_frames) >= self.threshold:
                # Find the maximum dimensions for padding
                max_height = max(frame.size(1) for frame in rgb_frames)
                max_width = max(frame.size(2) for frame in rgb_frames)

                # Pad the RGB frames and Depth frames
                rgb_frames_padded = [
                    F.pad(
                        frame,
                        (0, max_width - frame.size(2), 0, max_height - frame.size(1)),
                    )
                    for frame in rgb_frames
                ]
                depth_frames_padded = [
                    F.pad(
                        frame,
                        (0, max_width - frame.size(2), 0, max_height - frame.size(1)),
                    )
                    for frame in depth_frames
                ]

                # Stack the padded frames into tensors
                rgb_frames_tensor = torch.stack(rgb_frames_padded)
                depth_frames_tensor = torch.stack(depth_frames_padded)

                return Video(
                    rgb_frames=rgb_frames_tensor,
                    depth_frames=depth_frames_tensor,
                    original=label,
                )
        return None

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

    def calculate_depth_mask(self, face: np.ndarray) -> np.ndarray:
        """
        Calculates the depth mask for a given face image.

        Args:
            face (numpy.ndarray): The input face image as a NumPy array.

        Returns:
            numpy.ndarray: The depth mask as a NumPy array.

        """
        face = Image.fromarray(face)
        depth = self.pipeline(face)["depth"]
        depth = np.array(depth)
        depth = np.stack((depth,) * 3, axis=-1)
        return depth


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
        labels = []
        depth_frames = []
        rgb_frames = []
        for video in batch:
            # video.depth_frames = video.depth_frames.float() / 255.0
            video.depth_frames = self.repvit(video.depth_frames)
            video.depth_frames = self.positional_encoder(video.depth_frames)
            depth_frames.append(video.depth_frames)
            # video.rgb_frames = video.rgb_frames.float() / 255.0
            video.rgb_frames = self.repvit(video.rgb_frames)
            video.depth_frames = self.positional_encoder(video.rgb_frames)
            rgb_frames.append(video.rgb_frames)
            labels.append(int(video.original))
        depth_frames = torch.stack(depth_frames)
        rgb_frames = torch.stack(rgb_frames)
        labels = torch.tensor(labels)
        return rgb_frames, depth_frames, labels

    # The only purpose for this is for helping pylint with type annotations.
    def __iter__(self) -> Iterator[list[Video]]:
        return super().__iter__()


# if __name__ == "__main__":
#     from megatron.preprocessing import RepVit, PositionalEncoding

#     VIDEO_PATH = r"G:\My Drive\Megatron_DeepFake\dataset"
#     DEPTH_ANYTHING_SIZE = "Small"
#     NUM_FRAMES = 5
#     BATCH_SIZE = 2
#     SHUFFLE = True
#     dataset = VideoDataset(
#         VIDEO_PATH,
#         DEPTH_ANYTHING_SIZE,
#         num_frame=NUM_FRAMES,
#         num_video=BATCH_SIZE * 2,
#     )

#     dataloader = VideoDataLoader(
#         dataset,
#         RepVit(),
#         PositionalEncoding(384),
#         batch_size=BATCH_SIZE,
#         shuffle=SHUFFLE,
#     )
#     for batch in dataloader:
#         rgb_frames, depth_frames, labels = batch
#         print(rgb_frames.shape)
