"""Trainer class"""

from math import ceil
from os import PathLike
from typing import List, Literal

import torch
from torch import nn
import torch.optim as optim
from pydantic import BaseModel, Field
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from megatron import DEVICE
from megatron.trans_one import TransformerFakeDetector
from megatron.utils import save_checkpoint, save_model
from megatron.video_dataloader import Video, VideoDataLoader, VideoDataset
from megatron.preprocessing import RepVit, PositionalEncoding


class DatasetConfig(BaseModel):
    video_path: PathLike
    num_frames: int = Field(default=20)
    random_initial_frame: bool = Field(default=False)
    depth_anything_size: Literal["Small", "Base", "Large"] = Field(default="Small")


class DataloaderConfig(BaseModel):
    batch_size: int = Field(default=32)
    repvit_model: Literal[
        "repvit_m0_9.dist_300e_in1k",
        "repvit_m2_3.dist_300e_in1k",
        "repvit_m0_9.dist_300e_in1k",
        "repvit_m1_1.dist_300e_in1k",
        "repvit_m2_3.dist_450e_in1k",
        "repvit_m1_5.dist_300e_in1k",
        "repvit_m1.dist_in1k",
    ] = Field(default="repvit_m0_9.dist_300e_in1k")


class TransformerConfig(BaseModel):
    d_model: int = Field(default=384)
    n_heads: int = Field(default=2)
    n_layers: int = Field(default=1)
    d_ff: int = Field(default=1024)


class TrainConfig(BaseModel):
    learning_rate: float = Field(default=0.001)
    epochs: int = Field(default=1)
    log_dir: str


class Config(BaseModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    dataloader: DataloaderConfig = Field(default_factory=DataloaderConfig)
    transformer: TransformerConfig = Field(default_factory=TransformerConfig)
    train: TrainConfig


class Trainer:
    def __init__(self, config: Config, seed: torch.Generator):
        self.config = config
        self.model = TransformerFakeDetector(
            d_model=self.config.transformer.d_model,
            n_heads=self.config.transformer.n_heads,
            n_layers=self.config.transformer.n_layers,
            d_ff=self.config.transformer.d_ff,
            num_classes=2,
        ).to(DEVICE)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = (
            self.initialize_dataloader(seed)
        )
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config.train.learning_rate
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.writer = SummaryWriter(log_dir=self.config.train.log_dir)

    def initialize_dataloader(self, seed: torch.Generator):
        dataset = VideoDataset(
            video_dir=self.config.dataset.video_path,
            depth_anything_size=self.config.dataset.depth_anything_size,
            num_frame=self.config.dataset.num_frames,
            num_video=10,
        )
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size], generator=seed
        )
        repvit = RepVit().to(DEVICE)
        positional_encoder = PositionalEncoding(384).to(DEVICE)
        train_dataloader = VideoDataLoader(
            train_dataset,
            repvit,
            positional_encoder,
            batch_size=self.config.dataloader.batch_size,
            shuffle=True,
        )
        val_dataloader = VideoDataLoader(
            val_dataset,
            repvit,
            positional_encoder,
            batch_size=self.config.dataloader.batch_size,
            shuffle=True,
        )
        test_dataloader = VideoDataLoader(
            test_dataset,
            repvit,
            positional_encoder,
            batch_size=self.config.dataloader.batch_size,
            shuffle=True,
        )
        return train_dataloader, val_dataloader, test_dataloader

    def _train_step(self) -> float:
        self.model.train()
        train_loss = 0

        for batch in tqdm(
            self.train_dataloader,
            total=len(self.train_dataloader),
        ):

            rgb_frames, depth_frames, labels = batch
            rgb_frames = rgb_frames.to(DEVICE)
            depth_frames = depth_frames.to(DEVICE)
            labels = labels.to(DEVICE)
            _, loss = self.model(rgb_frames, depth_frames, labels)
            train_loss += loss.item()
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
            for batch in self.val_dataloader:
                rgb_frames, depth_frames, labels = batch
                rgb_frames = rgb_frames.to(DEVICE)
                depth_frames = depth_frames.to(DEVICE)
                labels = labels.to(DEVICE)
                _, loss = self.model(rgb_frames, depth_frames, labels)
                validation_loss += loss.item()
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
            validation_loss = self._validation_step()

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


if __name__ == "__main__":
    experiments = [
        {
            "dataset": {
                "video_path": r"G:\My Drive\Megatron_DeepFake\dataset",
                "num_frames": 10,
                "random_initial_frame": False,
                "depth_anything_size": "Small",
                "train_size": 0.5,
                "val_size": 0.3,
                "test_size": 0.2,
            },
            "dataloader": {
                "batch_size": 1,
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
                "epochs": 1,
                "log_dir": "data/runs/exp1",
                "early_stop_counter": 10,
            },
        }
    ]
    seed = torch.Generator().manual_seed(42)
    for experiment in experiments:
        trainer = Trainer(Config(**experiment), seed)
        trainer.train()
