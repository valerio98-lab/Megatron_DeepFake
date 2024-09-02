"""Trainer class"""

from math import ceil
from pathlib import Path
import transformers  # type: ignore
import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import Accuracy, F1Score

from tqdm.notebook import tqdm  # type: ignore

from megatron.configuration import ExperimentConfig
from megatron.preprocessing import PositionalEncoding, RepVit
from megatron.trans_one import TransformerFakeDetector
from megatron import utils
from megatron.video_dataloader import VideoDataLoader, VideoDataset


class Trainer:
    """
    Class used for performing experiments

    Attributes:
        device (torch.device): The device to be used for training.
        depth_anything (transformers.pipeline): The depth estimation model.
        repvit (RepVit): The RepVit model.
        positional_encoding (PositionalEncoding): The positional encoder.
        model (TransformerFakeDetector): The fake detector model.
        train_dataloader (VideoDataLoader): The training dataloader.
        val_dataloader (VideoDataLoader): The validation dataloader.
        test_dataloader (VideoDataLoader): The testing dataloader.
        criterion (nn.CrossEntropyLoss): The loss function.
        optimizer (optim.Adam): The optimizer.
        log_dir (Path): The directory to save logs.
        writer (tensorboard.SummaryWriter): The tensorboard writer.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initializes the Trainer class taking an experiment configuration.
        Args:
            config (ExperimentConfig): The experiment configuration.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_anything = transformers.pipeline(
            task="depth-estimation",
            model=f"depth-anything/Depth-Anything-V2-{config.dataset.depth_anything_size}-hf",
            device=self.device,
        )
        self.repvit = RepVit(repvit_model=config.dataloader.repvit_model).to(
            self.device
        )
        self.positional_encoding = PositionalEncoding(
            self.config.transformer.d_model, max_len=self.config.dataset.num_frames
        ).to(self.device)
        self.model = TransformerFakeDetector(
            d_model=self.config.transformer.d_model,
            n_heads=self.config.transformer.n_heads,
            n_layers=self.config.transformer.n_layers,
            d_ff=self.config.transformer.d_ff,
            num_classes=2,
            dropout=self.config.transformer.dropout,
        ).to(self.device)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = (
            self.initialize_dataloader()
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config.train.learning_rate
        )
        self.log_dir = Path(self.config.train.log_dir)
        self.writer = SummaryWriter(log_dir=self.config.train.log_dir)
        self.writer.add_text("Experiment info", (str(self.config)))

        self.accuracy = Accuracy(task='binary').to(self.device)
        self.f1_score = F1Score(task='multiclass', num_classes=2).to(self.device)

    def initialize_dataloader(self):
        """
        Initializes the training, validation, and testing dataloaders.
        Returns:
            tuple: A tuple containing the training, validation, and testing dataloaders.
        """
        dataset = VideoDataset(
            video_dir=self.config.dataset.video_path,
            depth_anything=self.depth_anything,
            num_frame=self.config.dataset.num_frames,
            num_video=self.config.dataset.num_video,
        )
        train_size = int(self.config.train.train_size * len(dataset))
        val_size = int(self.config.train.val_size * len(dataset))
        test_size = (
            len(dataset) - train_size - val_size
        )  # int(self.config.train.test_size * len(dataset))
        train_dataset, val_dataset, test_dataset = data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        dataloader_kwargs = {
            "repvit": self.repvit,
            "positional_encoding": self.positional_encoding,
            "batch_size": self.config.dataloader.batch_size,
            "shuffle": self.config.dataloader.shuffle,
            "pin_memory": self.config.dataloader.pin_memory,
            "num_workers": self.config.dataloader.num_workers,
        }
        train_dataloader = VideoDataLoader(train_dataset, **dataloader_kwargs)
        val_dataloader = VideoDataLoader(val_dataset, **dataloader_kwargs)
        test_dataloader = VideoDataLoader(test_dataset, **dataloader_kwargs)
        return train_dataloader, val_dataloader, test_dataloader

    def _train_step(self) -> float:

        self.model.train()
        train_loss = 0.0

        for batch in tqdm(
            iterable=self.train_dataloader,
            total=ceil(len(self.train_dataloader)),
            desc="TRAINING",
        ):
            rgb_frames, depth_frames, labels = batch
            logits = self.model(rgb_frames, depth_frames)
            loss = self.criterion(logits, labels)
            train_loss += loss.item()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        train_loss /= len(self.train_dataloader)
        return train_loss

    def _validation_step(self) -> float:

        self.model.eval()
        validation_loss = 0.0

        self.accuracy.reset()
        self.f1_score.reset()

        with torch.inference_mode():
            for batch in tqdm(
                iterable=self.val_dataloader,
                total=ceil(len(self.val_dataloader)),
                desc="VALIDATING",
            ):

                rgb_frames, depth_frames, labels = batch
                logits = self.model(rgb_frames, depth_frames)
                loss = self.criterion(logits, labels)
                validation_loss += loss.item()

                #Compute some metrics
                preds = torch.argmax(logits, dim=1)
                self.accuracy.update(preds, labels)
                self.f1_score.update(preds, labels)

                print("preds: ", preds)
                print("labels: ", labels)
        
        
        validation_loss /= len(self.val_dataloader)
        validation_accuracy = self.accuracy.compute()
        validation_f1_score = self.f1_score.compute()

        return validation_loss, validation_accuracy, validation_f1_score

    def train_and_validate(self):
        """
        Trains and validates the model.
        """
        print("N-heads: ", self.config.transformer.n_heads)
        print("layers: ", self.config.transformer.n_layers)
        print("d_ff: ", self.config.transformer.d_ff)
        initial_epoch = 0
        if self.config.train.resume_training:

            checkpoint_path = utils.get_latest_checkpoint(self.log_dir / "models")
            if checkpoint_path is not None:
                filename = checkpoint_path.stem
                initial_epoch = int(filename.split("_")[-1]) + 1
                utils.load_checkpoint(self.model, self.optimizer, checkpoint_path)
        if initial_epoch >= self.config.train.epochs:
            return
        for epoch in tqdm(
            range(initial_epoch, self.config.train.epochs),
            initial=initial_epoch,
            total=self.config.train.epochs,
            desc="Training and validating",
        ):

            # Training and validation steps
            train_loss = self._train_step()
            validation_loss, validation_accuracy, validation_f1_score = self._validation_step()

            print(f"\nEpoch: {epoch} ==> Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy}, Validation F1 Score: {validation_f1_score}\n")

            # Save checkpoint
            utils.save_checkpoint(self.model, epoch, self.optimizer, self.log_dir)

            self.writer.add_scalars(
                main_tag=f"Loss/{type(self.model).__name__}",
                tag_scalar_dict={
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                },
                global_step=epoch,
            )

            self.writer.add_scalar(
                f"Validation_Accuracy/{type(self.model).__name__}",
                validation_accuracy,
                global_step=epoch,
            )

            self.writer.add_scalar(
                f"Validation_F1-Score/{type(self.model).__name__}",
                validation_f1_score,
                global_step=epoch,
            )

        utils.save_model(self.model, self.log_dir)
        self.writer.flush()
        self.writer.close()

        return validation_loss

    def _test_step(self):

        test_loss = 0
        self.accuracy.reset()
        self.f1_score.reset()

        with torch.inference_mode():
            for batch in tqdm(
                iterable=self.val_dataloader,
                total=ceil(len(self.test_dataloader)),
                desc="TESTING",
            ):

                rgb_frames, depth_frames, labels = batch
                logits = self.model(rgb_frames, depth_frames)
                loss = self.criterion(logits, labels)
                test_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                self.accuracy.update(preds, labels)
                self.f1_score.update(preds, labels)


        test_loss /= len(self.val_dataloader)
        test_accuracy = self.accuracy.compute()
        test_f1_score = self.f1_score.compute()

        return test_loss, test_accuracy, test_f1_score

    def test(self):
        """
        Tests the trained model.
        """
        utils.load_model(self.model, self.log_dir)
        self.model.eval()
        test_loss, test_accuracy, test_f1_score = self._test_step()
        self.writer.add_scalars(
            main_tag=f"Loss/{type(self.model).__name__}",
            tag_scalar_dict={"test_loss": test_loss},
            global_step=0,
        )
        self.writer.add_scalar(
            f"Test_Accuracy/{type(self.model).__name__}",
            test_accuracy,
            global_step=0,
        )

        self.writer.add_scalar(
            f"Test_F1_Score/{type(self.model).__name__}",
            test_f1_score,
            global_step=0,
        )
        
        self.writer.flush()
        self.writer.close()
