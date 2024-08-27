from megatron.trans_one import TransformerFakeDetector
from megatron.utils import save_checkpoint, save_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from megatron.video_dataloader import VideoDataLoader, VideoDataset

from torch.utils.data import random_split
from tqdm import tqdm
from math import ceil

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = self.initialize_model()
        self.train_dataloader, self.val_dataloader, self.test_dataloader = (
            self.initialize_dataloader()
        )
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config["train"]["learning_rate"]
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.writer = SummaryWriter(log_dir=self.config["train"]["log_dir"])

    def initialize_dataloader(self):
        dataset = VideoDataset(
            video_dir=self.config["dataset"]["video_path"],
            depth_anything_size=self.config["dataset"]["depth_anything_size"],
            num_frame=self.config["dataset"]["num_frames"],
            num_video=20,
        )
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        train_dataloader = VideoDataLoader(
            train_dataset,
            batch_size=self.config["dataloader"]["batch_size"],
            shuffle=True,
        )
        val_dataloader = VideoDataLoader(
            val_dataset,
            batch_size=self.config["dataloader"]["batch_size"],
            shuffle=True,
        )
        test_dataloader = VideoDataLoader(
            test_dataset,
            batch_size=self.config["dataloader"]["batch_size"],
            shuffle=True,
        )
        return train_dataloader, val_dataloader, test_dataloader

    def initialize_model(self):
        model = TransformerFakeDetector(
            d_model=self.config["transformer"]["d_model"],
            n_heads=self.config["transformer"]["n_heads"],
            n_layers=self.config["transformer"]["n_layers"],
            d_ff=self.config["transformer"]["d_ff"],
            num_classes=2,
        )
        return model

    def _train_step(self) -> float:
        self.model.train()
        train_loss = 0

        for batch in tqdm(
            self.train_dataloader,
            total=ceil(len(self.train_dataloader) / self.train_dataloader.batch_size),
        ):
            # Move frames to device
            for video in batch:
                for frame in video.frames:
                    frame.depth_frame = frame.depth_frame.to(DEVICE)
                    frame.rgb_frame = frame.rgb_frame.to(DEVICE)

            # Forward pass
            _, loss = self.model(batch)
            train_loss += loss.item()

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Free up memory
            for video in batch:
                for frame in video.frames:
                    frame.depth_frame = frame.depth_frame.detach().cpu()
                    frame.rgb_frame = frame.rgb_frame.detach().cpu()

        train_loss /= len(self.train_dataloader)
        return train_loss

    def _validation_step(self) -> float:
        self.model.eval()
        validation_loss = 0

        with torch.inference_mode():
            for batch in tqdm(
                self.val_dataloader,
                total=ceil(len(self.val_dataloader) / self.val_dataloader.batch_size),
            ):
                # Move frames to device
                for video in batch:
                    for frame in video.frames:
                        frame.depth_frame = frame.depth_frame.to(DEVICE)
                        frame.rgb_frame = frame.rgb_frame.to(DEVICE)

                # Forward pass
                _, loss = self.model(batch)
                validation_loss += loss.item()

                # Free up memory
                for video in batch:
                    for frame in video.frames:
                        frame.depth_frame = frame.depth_frame.detach().cpu()
                        frame.rgb_frame = frame.rgb_frame.detach().cpu()

        validation_loss /= len(self.val_dataloader)
        return validation_loss

    def train(self):
        self.model.train()

        for epoch in tqdm(
            range(self.config["train"]["epochs"]),
            total=self.config["train"]["epochs"],
            desc="Training",
        ):
            # Training and validation steps
            train_loss = self._train_step()
            validation_loss = self._validation_step()

            # Save checkpoint
            print("SAVING CHECKPOINT...")
            save_checkpoint(
                self.model, epoch, self.optimizer, self.config["train"]["log_dir"]
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
        save_model(self.model, self.config["train"]["log_dir"])


if __name__ == "__main__":
    experiments = [
        {
            "dataset": {
                "video_path": r"G:\My Drive\Megatron_DeepFake\dataset",
                "num_frames": 20,
                "random_initial_frame": True,
                "depth_anything_size": "Small",
            },
            "dataloader": {
                "batch_size": 2,
                "repvit_model": "repvit_m0_9.dist_300e_in1k",
            },
            "transformer": {"d_model": 384, "n_heads": 2, "n_layers": 1, "d_ff": 1024},
            "train": {
                "learning_rate": 0.001,
                "epochs": 1,
                "log_dir": "data/runs/exp1",
                "early_stop_counter": 10,
            },
        }
    ]

    for config in experiments:
        trainer = Trainer(config)
        trainer.train()
