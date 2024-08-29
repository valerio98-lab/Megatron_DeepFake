"""Trainer class"""

from math import ceil
import transformers
import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.utils import tensorboard

from tqdm.autonotebook import tqdm

# from megatron import DEVICE
from megatron.configuration import Config
from megatron.preprocessing import PositionalEncoding, RepVit
from megatron.trans_one import TransformerFakeDetector
from megatron.utils import save_checkpoint, save_model
from megatron.video_dataloader import VideoDataLoader, VideoDataset


# TODO: Jose, crea la funzione di test
# TODO: Jose, Valerio, fare brainstorming insieme per capire la miglior combinazione di
# allocazione/deallocazione capire se conviene tenere tutto in ram a scapito di batch piu piccoli con meno frame
# o fare una gestione smart della memoria
class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.depth_anything = transformers.pipeline(
            task="depth-estimation",
            model=f"depth-anything/Depth-Anything-V2-{config.dataset.depth_anything_size}-hf",
            # TODO: Jose, Valerio, decommentare quando si rilascia,
            # in locale questo modello sembra sovraccaricara la VRAM.
            # device=DEVICE,
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
            dataset, [train_size, val_size, test_size]
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

        for batch in tqdm(
            iterable=self.train_dataloader,
            total=ceil(len(self.train_dataloader) / self.train_dataloader.batch_size),
            desc="TRAINING",
        ):
            rgb_frames, depth_frames, labels = self.load_data(batch)
            logits = self.model(rgb_frames, depth_frames, labels)
            loss = self.criterion(logits, labels)
            print(f"Nella train {train_loss=}")
            train_loss += loss.item()
            print(f"Nella train  {loss.item()=}")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            rgb_frames = rgb_frames.detach().cpu()
            depth_frames = depth_frames.detach().cpu()
            labels = labels.detach().cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        train_loss /= len(self.train_dataloader)
        return train_loss

    def _validation_step(self) -> float:
        self.model.eval()
        validation_loss = 0
        with torch.inference_mode():
            for batch in tqdm(
                iterable=self.val_dataloader,
                total=ceil(len(self.val_dataloader) / self.val_dataloader.batch_size),
                desc="VALIDATING",
            ):

                rgb_frames, depth_frames, labels = self.load_data(batch)
                logits = self.model(rgb_frames, depth_frames, labels)
                loss = self.criterion(logits, labels)
                print(f"Nella validation {validation_loss=}")
                validation_loss += loss.item()
                print(f"Nella validation  {loss.item()=}")
                rgb_frames = rgb_frames.detach().cpu()
                depth_frames = depth_frames.detach().cpu()
                labels = labels.detach().cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
        with torch.no_grad():
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
        return rgb_frames, depth_frames, labels


if __name__ == "__main__":
    import numpy as np

    experiment = {
        "dataset": {
            "video_path": r"H:\My Drive\Megatron_DeepFake\dataset",
            "num_frames": 1000,
            "random_initial_frame": True,
            "depth_anything_size": "Small",
            "num_video": 1,
            "frame_threshold": 10,
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
            "epochs": 2,
            "log_dir": "/data/runs/exp1",
            "early_stop_counter": 10,
            "train_size": 0.5,
            "val_size": 0.3,
            "test_size": 0.2,
        },
        "seed": 42,
    }

    DEVICE = "cpu"  # Avoid vram saturation
    config = Config(**experiment)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    trainer = Trainer(config)
    # TODO: Make it work
    for batch in trainer.train_dataloader:
        rgb_frames, depth_frames, labels = trainer.load_data(batch)
        print(f"{rgb_frames.shape=},{depth_frames.shape=},{labels=}")
    # trainer.train()
    # cnt_original = 0
    # cnt_manipulated = 0
    # for video_path in trainer.train_dataloader.dataset.dataset.video_paths:
    #     if "original" in video_path:
    #         cnt_original += 1
    #     elif "manipulated" in video_path:
    #         cnt_manipulated += 1
    # print(f"{cnt_original=}, {cnt_manipulated=}")
