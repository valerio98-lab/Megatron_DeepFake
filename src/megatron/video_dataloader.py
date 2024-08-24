"""Module containing the definition for the dasaet and 
dataloader"""

import os
import pathlib
from typing import Iterator, Literal
from dataclasses import dataclass

import cv2
import dlib
import numpy as np
import torch
import transformers
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import timm


@dataclass
class Frame:
    """
    Represents a frame containing an RGB frame and a depth frame.
    Attributes:
        rgb_frame (torch.Tensor): The RGB frame.
        depth_frame (torch.Tensor): The depth frame.
    """

    rgb_frame: torch.Tensor
    depth_frame: torch.Tensor

    def __repr__(self):
        return f"{type(self.rgb_frame)}, {type(self.depth_frame)}"


@dataclass
class Video:
    """
    Represents a video.
    Attributes:
        frames (list[Frame]): A list of frames in the video.
        original (bool): Indicates whether the video is original or not.
    """

    frames: list[Frame]
    original: bool

    def __repr__(self):
        return f"list[{str(self.frames[0])}], {type(self.original)}"


class VideoDataset(Dataset):
    """A PyTorch dataset for loading video data.
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
    Attributes:
        num_frame (int): The number of frames to extract from each video.
        threshold (int): The minimum number of frames required for a video to be considered valid.
        data_path (pathlib.Path): The path to the video directory.
        num_video (int | None): The maximum number of videos to load.
        random_initial_frame (bool): Whether to randomly select the initial frame for extraction.
        video_paths (list[str]): The list of video file paths.
        face_detector (dlib.fhog_object_detector): The face detector model.
        pipeline (transformers.pipelines.Pipeline): The depth estimation pipeline.
    Methods:
        __len__(): Returns the number of videos in the dataset.
        __collate_video(): Collates the video file paths from the directory.
        __getitem__(idx: int) -> Video | None: Retrieves a video and its corresponding label from the dataset.
        face_extraction(frame: np.ndarray) -> np.ndarray | None: Extracts the face from a frame image.
        calculate_depth_mask(face: np.ndarray) -> np.ndarray: Calculates the depth mask for a given face image.
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
                if cnt >= self.num_video and self.num_video is not None:
                    return video_paths
                if file.endswith(".mp4"):
                    cnt += 1
                    video_path = os.path.join(root, file)
                    video_paths.append(video_path)
        return video_paths

    def __getitem__(self, idx: int) -> Video | None:
        video_path = self.video_paths[idx]
        label = "manipulated" in video_path
        cap = cv2.VideoCapture(video_path)
        # get the number of frames in the video and set the length to the minimum between
        # the number of frames and the number of frames we want to extract.
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        length = min(total_frame, self.num_frame)
        if self.random_initial_frame:
            cap.set(
                cv2.CAP_PROP_POS_FRAMES, int(np.random.uniform(0, total_frame - length))
            )

        frames = []
        if cap.isOpened():
            for _ in range(length):
                # ret is a boolean value that indicates if the frame was read correctly or not.
                # frame is the image in BGR format.
                ret, frame = cap.read()

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
                face_crop = torch.from_numpy(face_crop)
                depth_mask = torch.from_numpy(depth_mask)

                frames.append(Frame(rgb_frame=face_crop, depth_frame=depth_mask))

            cap.release()
            if len(frames) >= self.threshold:
                return Video(frames=frames, original=label)
        return None

    def face_extraction(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Given a frame, if a face is found it returns the image cropped around the face,
        or else returns None.

        Parameters:
        - frame: np.ndarray
            The input frame containing the image.

        Returns:
        - np.ndarray | None
            If a face is found, it returns the cropped image around the face.
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

            video_dir: os.PathLike,
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
    Attributes:
        dataset (VideoDataset): The video dataset being loaded.
        repvit (RepVGG): The RepVGG model used for embedding extraction.
        batch_size (int): The batch size for loading the data.
        shuffle (bool): Whether to shuffle the data.
        collate_fn (callable): The collate function used for batching the data.
    Methods:
        __collate_fn(batch): A private method that performs collation on a batch of video data.
        get_repvit_embedding(img): Extracts the RepVGG embedding for an input image tensor.
        DataLoader(): Returns a DataLoader object for loading the video dataset.
    """

    def __init__(
        self,
        dataset: VideoDataset,
        repvit_model: Literal[
            "repvit_m0_9.dist_300e_in1k",
            "repvit_m2_3.dist_300e_in1k",
            "repvit_m0_9.dist_300e_in1k",
            "repvit_m1_1.dist_300e_in1k",
            "repvit_m2_3.dist_450e_in1k",
            "repvit_m1_5.dist_300e_in1k",
            "repvit_m1.dist_in1k",
        ] = "repvit_m0_9.dist_300e_in1k",
        batch_size=1,
        shuffle=True,
        custom_collate_fn=None,
    ):
        self.dataset = dataset
        self.repvit = timm.create_model(
            repvit_model,
            pretrained=True,
        ).eval()
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

    def __collate_fn(self, batch: list[Video]) -> list[Video]:
        for video in batch:
            for i, frame in enumerate(video.frames):
                rgb_frame = frame.rgb_frame.unsqueeze(0)
                depth_frame = frame.depth_frame.unsqueeze(0)
                with torch.no_grad():
                    embedded_rgb_frame = self.get_repvit_embedding(rgb_frame)
                    embedded_depth_frame = self.get_repvit_embedding(depth_frame)

                video.frames[i].rgb_frame = embedded_rgb_frame.squeeze(0)
                video.frames[i].depth_frame = embedded_depth_frame.squeeze(0)
        return batch

    def get_repvit_embedding(self, img: torch.Tensor) -> torch.Tensor:
        """
        Calculates the embeddings for the given image tensor using RepVit.

        Args:
            img (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The RepVit-embedding tensor for the input image.
        """
        img = img.float() / 255.0
        img = img.permute(0, 3, 1, 2)
        return self.repvit.forward_head(
            self.repvit.forward_features(img), pre_logits=True
        )

    # The only purpose for this is for helping pylint with type annotations.
    def __iter__(self) -> Iterator[list[Video]]:
        return super().__iter__()


# if __name__ == "__main__":

#     VIDEO_PATH = r"G:\My Drive\Megatron_DeepFake\dataset"
#     DEPTH_ANYTHING_SIZE = "Small"
#     NUM_FRAMES = 5
#     BATCH_SIZE = 2
#     SHUFFLE = True
#     dataset = VideoDataset(
#         VIDEO_PATH,
#         DEPTH_ANYTHING_SIZE,
#         num_frame=NUM_FRAMES,
#         num_video=BATCH_SIZE,
#     )
#     dataloader = VideoDataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
#     for batch in dataloader:
#         for video in batch:
#             for frame in video.frames:
#                 print(frame.rgb_frame.shape, frame.depth_frame.shape)
