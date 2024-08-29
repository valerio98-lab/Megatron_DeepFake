"""Trainer class"""

from os import PathLike
from typing import Literal

import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.utils import tensorboard

# from torch.utils.data import random_split
# from torch.utils.tensorboard import SummaryWriter
import transformers
from pydantic import BaseModel, Field, model_validator
from tqdm.autonotebook import tqdm

from megatron import DEVICE
from megatron.preprocessing import PositionalEncoding, RepVit
from megatron.trans_one import TransformerFakeDetector
from megatron.utils import save_checkpoint, save_model
from megatron.video_dataloader import VideoDataLoader, VideoDataset


class DatasetConfig(BaseModel):
    video_path: PathLike
    num_frames: int = Field(default=20)
    random_initial_frame: bool = Field(default=False)
    depth_anything_size: Literal["Small", "Base", "Large"] = Field(default="Small")
    num_video: int = Field(default=20)
    frame_threshold: int = Field(default=5)

    @model_validator(mode="after")
    def check_values(self):
        depth_anything_sizes = ["Small", "Base", "Large"]
        if self.depth_anything_size not in depth_anything_sizes:
            raise ValueError(
                f"depth_anything_size must be a value from {depth_anything_sizes}, but got {self.depth_anything_size}"
            )
        return self

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            f"DatasetConfig(video_path={self.video_path}, num_frames={self.num_frames}, "
            f"random_initial_frame={self.random_initial_frame}, depth_anything_size={self.depth_anything_size}, "
            f"num_video={self.num_video}, frame_threshold={self.frame_threshold})"
        )


class DataloaderConfig(BaseModel):
    batch_size: int = Field(default=32)
    repvit_model: str = Field(default="repvit_m0_9.dist_300e_in1k")

    @model_validator(mode="after")
    def check_values(self):
        repvit_models = [
            "repvit_m0_9.dist_300e_in1k",
            "repvit_m2_3.dist_300e_in1k",
            "repvit_m0_9.dist_300e_in1k",
            "repvit_m1_1.dist_300e_in1k",
            "repvit_m2_3.dist_450e_in1k",
            "repvit_m1_5.dist_300e_in1k",
            "repvit_m1.dist_in1k",
        ]
        if self.repvit_model not in repvit_models:
            raise ValueError(
                f"repvit_model must be a value from {repvit_models}, but got {self.repvit_model}"
            )
        return self

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"DataloaderConfig(batch_size={self.batch_size}, repvit_model={self.repvit_model})"


class TransformerConfig(BaseModel):
    d_model: int = Field(default=384)
    n_heads: int = Field(default=2)
    n_layers: int = Field(default=1)
    d_ff: int = Field(default=1024)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            f"TransformerConfig(d_model={self.d_model}, n_heads={self.n_heads}, "
            f"n_layers={self.n_layers}, d_ff={self.d_ff})"
        )


class TrainConfig(BaseModel):
    learning_rate: float = Field(default=0.001)
    epochs: int = Field(default=1)
    log_dir: str
    early_stop_counter: int = Field(default=10)
    train_size: float = Field(default=0.6)
    val_size: float = Field(default=0.3)
    test_size: float = Field(default=0.1)
    seed: int = Field(default=42)

    @model_validator(mode="after")
    def check_values(self):
        total = self.train_size + self.val_size + self.test_size
        if total != 1.0:
            raise ValueError(
                f"train_size, val_size, and test_size must sum up to 1.0, but got {total}"
            )
        return self

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            f"TrainConfig(learning_rate={self.learning_rate}, epochs={self.epochs}, log_dir={self.log_dir}, "
            f"early_stop_counter={self.early_stop_counter}, train_size={self.train_size}, "
            f"val_size={self.val_size}, test_size={self.test_size}, seed={self.seed})"
        )


class Config(BaseModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    dataloader: DataloaderConfig = Field(default_factory=DataloaderConfig)
    transformer: TransformerConfig = Field(default_factory=TransformerConfig)
    train: TrainConfig

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            f"Config(\n  dataset={self.dataset},\n  dataloader={self.dataloader},\n  "
            f"transformer={self.transformer},\n  train={self.train}\n)"
        )


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.depth_anything = transformers.pipeline(
            task="depth-estimation",
            model=f"depth-anything/Depth-Anything-V2-{config.dataset.depth_anything_size}-hf",
            device=DEVICE,
        )
        self.repvit = RepVit(config.dataloader.repvit_model).to(DEVICE)
        self.positional_encoder = PositionalEncoding(
            self.config.transformer.d_model
        ).to(DEVICE)
        self.model = TransformerFakeDetector(
            d_model=self.config.transformer.d_model,
            n_heads=self.config.transformer.n_heads,
            n_layers=self.config.transformer.n_layers,
            d_ff=self.config.transformer.d_ff,
            num_classes=2,
        ).to(DEVICE)
        self.generator = torch.Generator().manual_seed(self.config.train.seed)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = (
            self.initialize_dataloader()
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config.train.learning_rate
        )
        self.writer = tensorboard.SummaryWriter(log_dir=self.config.train.log_dir)
        self.writer.add_text("Experiment info", (str(self.config)))

    def initialize_dataloader(self):
        dataset = VideoDataset(
            video_dir=self.config.dataset.video_path,
            depth_anything=self.depth_anything,
            num_frame=self.config.dataset.num_frames,
            num_video=self.config.dataset.num_video,
        )
        train_size = int(self.config.train.train_size * len(dataset))
        val_size = int(self.config.train.val_size * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = data.random_split(
            dataset, [train_size, val_size, test_size], generator=self.generator
        )
        train_dataloader = VideoDataLoader(
            train_dataset,
            self.repvit,
            self.positional_encoder,
            batch_size=self.config.dataloader.batch_size,
            shuffle=True,
        )
        val_dataloader = VideoDataLoader(
            val_dataset,
            self.repvit,
            self.positional_encoder,
            batch_size=self.config.dataloader.batch_size,
            shuffle=True,
        )
        test_dataloader = VideoDataLoader(
            test_dataset,
            self.repvit,
            self.positional_encoder,
            batch_size=self.config.dataloader.batch_size,
            shuffle=True,
        )
        return train_dataloader, val_dataloader, test_dataloader

    def _train_step(self) -> float:
        self.model.train()
        train_loss = 0
        print(f"{len(self.train_dataloader)=}, {self.train_dataloader.batch_size=}")
        for batch in tqdm(
            self.train_dataloader,
            total=(len(self.train_dataloader) / self.train_dataloader.batch_size),
            desc="TRAINING",
        ):
            rgb_frames, depth_frames, labels = self.load_data(batch)
            logits, _ = self.model(rgb_frames, depth_frames, labels)
            print(f"{logits.shape=}")
            loss = self.criterion(logits, labels)
            train_loss += loss.item()
            print(f"Nel train, bce,  {loss.item()=}")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            rgb_frames = rgb_frames.detach().cpu()
            depth_frames = depth_frames.detach().cpu()
            labels = labels.detach().cpu()
        train_loss /= len(self.train_dataloader)
        return train_loss

    def _validation_step(self) -> float:
        self.model.eval()
        validation_loss = 0
        with torch.inference_mode():
            for batch in tqdm(
                self.val_dataloader,
                total=len(self.val_dataloader) / self.val_dataloader.batch_size,
                desc="VALIDATING",
            ):

                rgb_frames, depth_frames, labels = self.load_data(batch)
                logits, _ = self.model(rgb_frames, depth_frames, labels)
                loss = self.criterion(logits, labels)
                print(f"Nella validation {validation_loss=}")
                validation_loss += loss.item()
                print(f"Nella validation  {loss.item()=}")
                rgb_frames = rgb_frames.detach().cpu()
                depth_frames = depth_frames.detach().cpu()
                labels = labels.detach().cpu()
        validation_loss /= len(self.val_dataloader)

        return validation_loss

    def train(self):
        self.model.train()
        for epoch in tqdm(
            range(self.config.train.epochs),
            total=self.config.train.epochs,
            desc="Training and validating",
        ):
            # Training and validation steps
            train_loss = self._train_step()

            print("EXIT TRAIN...")
            validation_loss = self._validation_step()

            print("EXIT VALIDATION...")

            # Save checkpoint
            print("SAVING CHECKPOINT...")
            save_checkpoint(
                self.model, epoch, self.optimizer, self.config.train.log_dir
            )

            # Log to TensorBoard
            print("SAVING RUN FOR TENSORBOARD...")
            self.writer.add_scalars(
                main_tag=f"Loss_{str(type(self.model).__name__)}",
                tag_scalar_dict={
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                },
                global_step=epoch,
            )

        self.writer.flush()
        self.writer.close()

        # Save final model
        save_model(self.model, self.config.train.log_dir)

    def load_data(self, batch):
        labels = []
        depth_frames = []
        rgb_frames = []
        for video in batch:
            video.depth_frames = self.repvit(video.depth_frames.to(DEVICE))
            video.depth_frames = self.positional_encoder(video.depth_frames)
            depth_frames.append(video.depth_frames)

            video.rgb_frames = self.repvit(video.rgb_frames.to(DEVICE))
            video.rgb_frames = self.positional_encoder(video.rgb_frames)
            rgb_frames.append(video.rgb_frames)
            labels.append(int(video.original))

        depth_frames = torch.stack(depth_frames)
        rgb_frames = torch.stack(rgb_frames)
        labels = torch.tensor(labels).to(DEVICE)
        # print(f"{depth_frames.shape=},{rgb_frames.shape=},{labels.shape=}")
        return rgb_frames, depth_frames, labels


# if __name__ == "__main__":
#     experiments = [
#         {
#             "dataset": {
#                 "video_path": r"H:\My Drive\Megatron_DeepFake\dataset",
#                 "num_frames": 10,
#                 "random_initial_frame": False,
#                 "depth_anything_size": "Small",
#                 "num_video": 10,
#                 "frame_threshold": 10,
#             },
#             "dataloader": {
#                 "batch_size": 1,
#                 "repvit_model": "repvit_m0_9.dist_300e_in1k",
#             },
#             "transformer": {
#                 "d_model": 384,
#                 "n_heads": 2,
#                 "n_layers": 1,
#                 "d_ff": 1024,
#             },
#             "train": {
#                 "learning_rate": 0.001,
#                 "epochs": 1,
#                 "log_dir": "data/runs/exp1",
#                 "early_stop_counter": 10,
#                 "train_size": 0.5,
#                 "val_size": 0.3,
#                 "test_size": 0.2,
#                 "seed": 42,
#             },
#         }
#     ]
#     for experiment in experiments:
#         trainer = Trainer(Config(**experiment))
#         trainer.train()
