from os import PathLike

import torch
from torch import nn
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from megatron.trans_one import TransformerFakeDetector
from megatron.utils import save_checkpoint, save_model
from megatron.video_dataloader import VideoDataLoader, VideoDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(
        self,
        fake_detector: TransformerFakeDetector,
        train_dataloader: VideoDataLoader,
        val_dataloader: VideoDataLoader,
        test_dataloder: VideoDataLoader,
        save_path: PathLike,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        epochs: int = 100,
        early_stopping_range: int = 10,
    ) -> None:
        self.model = fake_detector.to(device=DEVICE)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloder = test_dataloder
        self.save_path = save_path
        self.optimizer = optimizer
        self.epochs = epochs
        self.early_stopping_range = early_stopping_range

    def _train_step(self) -> float:
        self.model.train()
        train_loss = 0

        for batch in tqdm(
            self.train_dataloader,
            total=ceil(len(self.train_dataloader) / self.train_dataloader.batch_size),
        ):

            for video in batch:
                for frame in video.frames:
                    frame.depth_frame.to(DEVICE)
                    frame.rgb_frame.to(DEVICE)

            _, loss = self.model(batch)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Free up memory
            for video in batch:
                for frame in video.frames:
                    frame.depth_frame.detach().cpu()
                    frame.rgb_frame.detach().cpu()
        train_loss /= len(self.train_dataloader)
        return train_loss

    def _validation_step(self) -> float:
        self.model.eval()
        validation_loss = 0
        with torch.inference_mode():
            for batch in self.val_dataloader:
                for video in batch:
                    for frame in video.frames:
                        frame.depth_frame.to(DEVICE)
                        frame.rgb_frame.to(DEVICE)
                _, loss = self.model(batch)
                validation_loss += loss.item()
                for video in batch:
                    for frame in video.frames:
                        frame.depth_frame.detach().cpu()
                        frame.rgb_frame.detach().cpu()
        validation_loss /= len(self.val_dataloader)

        return validation_loss

    def train(
        self,
    ):
        self.model.train()
        writer = SummaryWriter(
            self.save_path,
            comment="test",  # Commento per differenziare gli esperimenti
        )
        best_val_loss = float("inf")
        early_stop_counter = 0
        for epoch in tqdm(range(self.epochs), total=self.epochs, desc="Training"):
            train_loss = self._train_step()
            validation_loss = self._validation_step()
            print("SAVING CHECKPOINT...")
            save_checkpoint(
                self.model,
                epoch,
                self.optimizer,
                self.save_path,
            )
            print("SAVING RUN FOR TENSORBOARD...")
            writer.add_scalars(
                main_tag=f"Loss_{str(type(self.model).__name__)}",
                tag_scalar_dict={
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                },
                global_step=epoch,
            )

            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.early_stopping_range:
                print(f"Early stopping training at epoch: {epoch+1}")
                break

        writer.flush()
        writer.close()
        save_model(self.model, self.save_path)


if __name__ == "__main__":
    VIDEO_PATH = r"G:\My Drive\Megatron_DeepFake\dataset"
    DEPTH_ANYTHING_SIZE = "Small"
    NUM_FRAMES = 5
    BATCH_SIZE = 2
    SHUFFLE = True
    dataset = VideoDataset(
        VIDEO_PATH, DEPTH_ANYTHING_SIZE, num_frame=NUM_FRAMES, num_video=20
    )

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    train_dataloader = VideoDataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE
    )
    val_dataloader = VideoDataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE
    )
    test_dataloader = VideoDataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE
    )
    fake_detector = TransformerFakeDetector(384, 2, 1, 1024, 2)
    trainer = Trainer(
        fake_detector,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        "./data/",
        optimizer=torch.optim.Adam(fake_detector.parameters(), lr=0.0001),
        epochs=1,
    )
    trainer.train()
