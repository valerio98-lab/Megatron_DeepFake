"""Trainer class"""

import os
from math import ceil
from pathlib import Path
import json
from typing import Optional
import transformers  # type: ignore
import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.utils.tensorboard.writer import SummaryWriter

# from torchmetrics import Accuracy, F1Score # sorry valerio but mypy won.
from torchmetrics.classification import BinaryAccuracy, MulticlassF1Score

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
            batch_size=self.config.dataset.num_frames,
        )
        self.repvit = RepVit(repvit_model=config.dataloader.repvit_model).to(
            self.device
        )
        self.positional_encoding = PositionalEncoding(
            384, max_len=self.config.dataset.num_frames
        ).to(self.device)
        self.model = TransformerFakeDetector(
            d_model=self.config.transformer.d_model,
            n_heads=self.config.transformer.n_heads,
            n_layers=self.config.transformer.n_layers,
            d_ff=self.config.transformer.d_ff,
            num_classes=2,
            projector_bool=self.config.transformer.projector_bool,
            dropout=self.config.transformer.dropout,
            pooling_type=self.config.transformer.pooling_type,
            activation=self.config.transformer.activation,
        ).to(self.device)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = (
            self.initialize_dataloader()
        )
        optim_kwargs = {
            "params": self.model.parameters(),
            "lr": config.train.learning_rate,
            "weight_decay": config.train.weight_decay,
        }
        self.criterion = nn.CrossEntropyLoss()
        if self.config.train.optim == "adam":
            self.optimizer = optim.Adam(**optim_kwargs)
        elif self.config.train.optim == "sgd":
            self.optimizer = optim.SGD(**optim_kwargs)
        elif self.config.train.optim == "rmsprop":
            self.optimizer = optim.RMSprop(**optim_kwargs)
        else:
            raise ValueError("Unsupported optimizer")

        # self.optimizer = self.optimizer(
        #     self.model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay
        # )
        self.log_dir = Path(self.config.train.log_dir)

        if self.config.techniques is not None:
            self.tmp = (
                Path(self.config.train.tmp_dir)
                / "_".join(self.config.techniques)
                / f"batch_size_{self.config.dataloader.batch_size}"
                / f"num_frames_{self.config.dataset.num_frames}"
                / f"depth_anything_size_{self.config.dataset.depth_anything_size}"
                / f"repvit_model_{self.config.dataloader.repvit_model}".replace(".", "")
                / "d_model_384"
            )
        else:
            self.tmp = (
                Path(self.config.train.tmp_dir)
                / "all"
                / f"batch_size_{self.config.dataloader.batch_size}"
                / f"num_frames_{self.config.dataset.num_frames}"
                / f"depth_anything_size_{self.config.dataset.depth_anything_size}"
                / f"repvit_model_{self.config.dataloader.repvit_model}".replace(".", "")
                / "d_model_384"
            )
        self.accuracy: BinaryAccuracy = BinaryAccuracy().to(self.device)
        self.f1_score: MulticlassF1Score = MulticlassF1Score(num_classes=2).to(
            self.device
        )

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
            techniques=self.config.techniques,
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

    def _validation_step(
        self,
    ) -> tuple[float, torch.Tensor, torch.Tensor]:  # type :ignore

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

                # Compute some metrics
                preds = torch.argmax(logits, dim=1)
                self.accuracy.update(preds, labels)
                self.f1_score.update(preds, labels)

        validation_loss /= len(self.val_dataloader)
        validation_accuracy = self.accuracy.compute()  # type :ignore
        validation_f1_score = self.f1_score.compute()  # type :ignore

        return validation_loss, validation_accuracy, validation_f1_score  # type :ignore

    def train_and_validate(self):
        """
        Trains and validates the model.
        """
        writer = SummaryWriter(log_dir=self.config.train.log_dir)
        writer.add_text("Experiment info", (str(self.config)))
        initial_epoch = self.resume_training_if_possible()
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
            validation_loss, validation_accuracy, validation_f1_score = (
                self._validation_step()
            )

            print(
                f"\nEpoch: {epoch} ==> {validation_loss=}, {validation_accuracy=}, {validation_f1_score=}\n"
            )
            writer.add_scalars(
                f"Loss/{type(self.model).__name__}",
                {"train_loss": train_loss, "validation_loss": validation_loss},
                epoch,
            )
            writer.add_scalar(
                f"Validation_Accuracy/{type(self.model).__name__}",
                validation_accuracy,
                epoch,
            )
            writer.add_scalar(
                f"Validation_F1-Score/{type(self.model).__name__}",
                validation_f1_score,
                epoch,
            )
            # Save checkpoint
            utils.save_checkpoint(self.model, epoch, self.optimizer, self.log_dir)

        utils.save_model(self.model, self.log_dir)
        writer.flush()
        writer.close()

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
        writer = SummaryWriter(log_dir=self.config.train.log_dir)
        writer.add_text("Experiment info", (str(self.config)))
        utils.load_model(self.model, self.log_dir)
        self.model.eval()
        test_loss, test_accuracy, test_f1_score = self._test_step()
        writer.add_scalars(
            main_tag=f"Loss/{type(self.model).__name__}",
            tag_scalar_dict={"test_loss": test_loss},
            global_step=0,
        )
        writer.add_scalar(
            f"Test_Accuracy/{type(self.model).__name__}",
            test_accuracy,
            global_step=0,
        )

        writer.add_scalar(
            f"Test_F1_Score/{type(self.model).__name__}",
            test_f1_score,
            global_step=0,
        )

        writer.flush()
        writer.close()

    def _optimized_train_step(
        self, rgb_frames_train_files, depth_frames_train_files, labels_train_files
    ) -> float:

        self.model.train()
        train_loss = 0.0
        for rgb_frames_train_file, depth_frames_train_file, labels_train_file in tqdm(
            iterable=zip(
                rgb_frames_train_files, depth_frames_train_files, labels_train_files
            ),
            total=len(rgb_frames_train_files),
            desc="TRAINING",
            disable=True,
        ):

            rgb_frames = torch.load(rgb_frames_train_file, weights_only=True)
            depth_frames = torch.load(depth_frames_train_file, weights_only=True)
            labels = torch.load(labels_train_file, weights_only=True)
            logits = self.model(rgb_frames, depth_frames)
            loss = self.criterion(logits, labels)
            train_loss += loss.item()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        train_loss /= len(self.train_dataloader)
        return train_loss

    def _optimized_validation_step(
        self, rgb_frames_val_files, depth_frames_val_files, labels_val_files
    ) -> tuple[float, torch.Tensor, torch.Tensor]:

        self.model.eval()
        validation_loss = 0.0
        self.accuracy.reset()
        self.f1_score.reset()
        with torch.inference_mode():
            for rgb_frames_val_file, depth_frames_val_file, labels_val_file in tqdm(
                iterable=zip(
                    rgb_frames_val_files, depth_frames_val_files, labels_val_files
                ),
                total=len(rgb_frames_val_files),
                desc="VALIDATING",
                disable=True,
            ):
                rgb_frames = torch.load(rgb_frames_val_file, weights_only=True)
                depth_frames = torch.load(depth_frames_val_file, weights_only=True)
                labels = torch.load(labels_val_file, weights_only=True)
                logits = self.model(rgb_frames, depth_frames)
                loss = self.criterion(logits, labels)
                validation_loss += loss.item()

                # Compute some metrics
                preds = torch.argmax(logits, dim=1)
                self.accuracy.update(preds, labels)
                self.f1_score.update(preds, labels)
        validation_loss /= len(self.val_dataloader)
        validation_accuracy = self.accuracy.compute()
        validation_f1_score = self.f1_score.compute()
        return validation_loss, validation_accuracy, validation_f1_score  # type :ignore

    def optimized_train_and_validate(self):
        """
        Trains and validates the model.
        """
        writer = SummaryWriter(log_dir=self.config.train.log_dir)
        writer.add_text("Experiment info", (str(self.config)))

        initial_epoch = self.resume_training_if_possible()
        if initial_epoch >= self.config.train.epochs:
            return

        training_files, validation_files = self.cache_data_if_needed()

        for epoch in tqdm(
            range(initial_epoch, self.config.train.epochs),
            initial=initial_epoch,
            total=self.config.train.epochs,
            desc="Training and validating",
        ):

            # Training and validation steps
            train_loss = self._optimized_train_step(*training_files)
            validation_loss, validation_accuracy, validation_f1_score = (
                self._optimized_validation_step(*validation_files)
            )

            # print(
            #     f"\nEpoch: {epoch} ==> {validation_loss=}, {validation_accuracy=}, {validation_f1_score=}\n"
            # )
            writer.add_scalars(
                f"Loss/{type(self.model).__name__}",
                {"train_loss": train_loss, "validation_loss": validation_loss},
                epoch,
            )
            writer.add_scalar(
                f"Validation_Accuracy/{type(self.model).__name__}",
                validation_accuracy,
                epoch,
            )
            writer.add_scalar(
                f"Validation_F1-Score/{type(self.model).__name__}",
                validation_f1_score,
                epoch,
            )
            # Save checkpoint
            utils.save_checkpoint(self.model, epoch, self.optimizer, self.log_dir)

        utils.save_model(self.model, self.log_dir)
        writer.flush()
        writer.close()

        return validation_loss, validation_f1_score

    def resume_training_if_possible(self):
        """
        Resume training from the last checkpoint if available.
        """
        initial_epoch = 0
        if self.config.train.resume_training:
            checkpoint_path = utils.get_latest_checkpoint(self.log_dir / "models")
            if checkpoint_path:
                filename = checkpoint_path.stem
                initial_epoch = int(filename.split("_")[-1]) + 1
                utils.load_checkpoint(self.model, self.optimizer, checkpoint_path)
        return initial_epoch

    def cache_data_if_needed(self):
        """
        Cache data for training and validation if it has not been cached already.
        """
        rgb_frames_train_files, depth_frames_train_files, labels_train_files = (
            [],
            [],
            [],
        )
        rgb_frames_val_files, depth_frames_val_files, labels_val_files = [], [], []

        os.makedirs(self.tmp, exist_ok=True)
        self._cache_data(
            self.train_dataloader,
            "train",
            rgb_frames_train_files,
            depth_frames_train_files,
            labels_train_files,
        )
        self._cache_data(
            self.val_dataloader,
            "val",
            rgb_frames_val_files,
            depth_frames_val_files,
            labels_val_files,
        )

        return (rgb_frames_train_files, depth_frames_train_files, labels_train_files), (
            rgb_frames_val_files,
            depth_frames_val_files,
            labels_val_files,
        )

    def _split(
        self, t: torch.Tensor, idx: int
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Takes a tensor t and an index idx, and splits t into two parts based on the first dimension."""
        if idx == 0:
            return None, t
        if t.shape[0] > idx:
            return t[:idx], t[idx:]
        return t, None

    def _load_cached_data(self, prefix, rgb_files, depth_files, label_files):
        state_path = self.tmp / f"{prefix}_state.json"
        batch_index = 0
        dataloader_start_index = 0
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as state:
                state_dict = json.load(state)
                batch_index = min(value["batch_index"] for value in state_dict.values())
                dataloader_start_index = min(
                    value["dataloader_index"] for value in state_dict.values()
                )
        filenames = [
            f"{prefix}_rgb_batch_{{}}",
            f"{prefix}_depth_batch_{{}}",
            f"{prefix}_labels_batch_{{}}",
        ]
        return_lists = [rgb_files, depth_files, label_files]
        for batch in range(batch_index):
            for idx, (filename, _) in enumerate(zip(filenames, return_lists)):
                filename = self.tmp / filename.format(batch)
                return_lists[idx].append(filename)
        return batch_index, dataloader_start_index

    def _cache_data(self, dataloader, prefix, rgb_files, depth_files, label_files):
        """
        Cache data batches to disk and updates the list in place

        """
        state_path = self.tmp / f"{prefix}_state.json"

        batch_index, dataloader_start_index = self._load_cached_data(
            prefix, rgb_files, depth_files, label_files
        )

        return_lists = [rgb_files, depth_files, label_files]
        accumulators = [None, None, None]
        filenames = [
            f"{prefix}_rgb_batch_{{}}",
            f"{prefix}_depth_batch_{{}}",
            f"{prefix}_labels_batch_{{}}",
        ]

        for dataloader_index, batch in tqdm(
            enumerate(
                dataloader[dataloader_start_index:], start=dataloader_start_index
            ),
            initial=dataloader_start_index,
            total=len(dataloader),
            desc=f"Caching {prefix} data",
        ):
            current_state = {}
            for idx, (accumulator, batch_data) in enumerate(zip(accumulators, batch)):

                if accumulator is not None:
                    batch_data = torch.cat((accumulator, batch_data))
                if batch_data.shape[0] >= dataloader.batch_size:
                    save_data, batch_data = (
                        batch_data[: dataloader.batch_size],
                        batch_data[dataloader.batch_size :],
                    )
                    filename = self.tmp / filenames[idx].format(batch_index)
                    torch.save(save_data, filename)
                    return_lists[idx].append(filename)
                    current_state[str(filename)] = {
                        "batch_index": batch_index,
                        "dataloader_index": dataloader_index,
                    }
                    if batch_data.shape[0] == 0:
                        batch_data = None
                accumulators[idx] = batch_data
            if current_state:
                with open(state_path, "w", encoding="utf-8") as state:
                    json.dump(current_state, state)
                    batch_index += 1

        current_state = {}
        for idx, accumulator in enumerate(accumulators):
            if accumulator is not None:
                filename = self.tmp / filenames[idx].format(batch_index)
                return_lists[idx].append(filename)
                torch.save(accumulator, filename)
                current_state[str(filename)] = {
                    "batch_index": batch_index,
                    "dataloader_index": len(dataloader),
                }
        if current_state:
            with open(state_path, "w", encoding="utf-8") as state:
                json.dump(current_state, state)
