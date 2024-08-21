"""Module containing the definition for the dasaet and 
dataloader"""

import os
import pathlib
from typing import Literal
import cv2
import dlib
import numpy as np
import torch
import transformers
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import timm


class VideoDataset(Dataset):
    """
    Initializes the dataset class.
    Args:
        - video_dir (os.PathLike): The directory path where the video files are located.
        - depth_anything_size (Literal["Small", "Base", "Large"], optional): The size of the depth-anything model to use.
            Defaults to "Small".
        - num_video (int | None, optional): The number of videos to load.
            If None, all videos in the directory will be loaded. Defaults to None.
        - threshold (int, optional): The threshold value for face detection. Defaults to 5.
        - num_frame (int, optional): The number of frames to extract from each video.
            Defaults to 1.
        - random_initial_frame (bool, optional): Whether to randomly select the initial frame for each video.
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

        self.video_files = []
        self.face_detector = dlib.get_frontal_face_detector()
        self.pipeline = transformers.pipeline(
            task="depth-estimation",
            model=f"depth-anything/Depth-Anything-V2-{depth_anything_size}-hf",
        )

        self.video_files = self.__collate_video()

    def __len__(self) -> int:
        return len(self.video_files)

    def __collate_video(self) -> list[(str, bool)]:
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

    def __getitem__(self, idx: int) -> np.ndarray | None:
        video_path, label = self.video_files[idx]
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

                frames.append((face_crop, depth_mask, label))

            cap.release()
            if len(frames) >= self.threshold:
                return frames
        return None

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
        depth = np.stack((depth,) * 3, axis=-1)
        return depth


class VideoDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
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

    def __collate_fn(self, batch):
        embedded_batch = []
        for video in batch:
            embedded_frames = []
            for rgb_crop, depth_mask, label in video:
                rgb_crop = rgb_crop.unsqueeze(0)
                print(rgb_crop.shape)
                depth_mask = depth_mask.unsqueeze(0)
                print(depth_mask.shape)

                with torch.no_grad():
                    embedded_rgb = self.get_repvit_embedding(rgb_crop)
                    embedded_depth = self.get_repvit_embedding(depth_mask)

                embedded_frames.append((embedded_rgb, embedded_depth, label))
            embedded_batch.append(embedded_frames)

        return embedded_batch

    def get_repvit_embedding(self, img: torch.Tensor):
        img = img.float() / 255.0
        print(img.shape)
        img = img.permute(0, 3, 1, 2)
        return self.repvit.forward_head(
            self.repvit.forward_features(img), pre_logits=True
        )

    def DataLoader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
        )


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
#         print(len(batch))
#         for elem in batch:
#             for x in elem:
#                 print(len(x), end=" ")
#                 # fig, axs = plt.subplots(2, 2, figsize=(10, 10))
#                 # plt.title("original" if x[2] else "manipulated")
#                 print(type(x[0]), type(x[1]), type(x[2]))
