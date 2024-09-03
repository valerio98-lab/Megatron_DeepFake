"""This module contains the necessary classes for configuring each experiment"""

from os import PathLike
from pydantic import BaseModel, Field, model_validator
from typing import Optional


class DatasetConfig(BaseModel):
    """
    Configuration class for dataset settings.
    Attributes:
        video_path (PathLike): The path to the video.
        num_frames (int, optional): The number of frames to extract from the video. Defaults to 20.
        random_initial_frame (bool, optional): Whether to randomly select the initial frame. Defaults to False.
        depth_anything_size (str, optional): The size of the depth anything.
            Must be one of ["Small", "Base", "Large"]. Defaults to "Small".
        num_video (int, optional): The number of videos. Defaults to 20.
    """

    video_path: PathLike
    num_frames: int = Field(default=20)
    random_initial_frame: bool = Field(default=False)
    depth_anything_size: str = Field(default="Small")
    num_video: int = Field(default=20)

    @model_validator(mode="after")
    def check_values(self):
        """
        Validates the values of the configuration attributes.

        Raises:
            ValueError: If `depth_anything_size` is not a valid value.

        Returns:
            Configuration: The updated configuration object.
        """

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
            f"num_video={self.num_video})"
        )


class DataloaderConfig(BaseModel):
    """
    Configuration class for dataloader settings.
    Attributes:
        batch_size (int, optional): The batch size. Defaults to 32.
        repvit_model (str, optional): The repvit model. Must be one of ["repvit_m0_9.dist_300e_in1k",
            "repvit_m2_3.dist_300e_in1k", "repvit_m0_9.dist_300e_in1k", "repvit_m1_1.dist_300e_in1k",
            "repvit_m2_3.dist_450e_in1k", "repvit_m1_5.dist_300e_in1k", "repvit_m1.dist_in1k"].
            Defaults to "repvit_m0_9.dist_300e_in1k".
        shuffle (bool, optional): Wether or not to shuffle the batches.
        pin_memory (bool, optional): Wether or not pin memory, for more informations
            look at this [discussion](https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723/4).
            Defaults to True.
        num_workers (int, optional): Wether or not use more workers, for more informations
            look at this
            [discussion](https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813).
            Defaults to 4.
    """

    batch_size: int = Field(default=32)
    # cached_batch_size: int = Field(default=8)
    shuffle: bool = Field(default=True)
    repvit_model: str = Field(default="repvit_m0_9.dist_300e_in1k")
    pin_memory: bool = Field(default=True)
    num_workers: int = Field(default=True)

    @model_validator(mode="after")
    def check_values(self):
        """
        Check the values of the `repvit_model` attribute.

        This method is a model validator that checks if the `repvit_model` attribute is a valid value.
        It compares the value of `repvit_model` with a list of valid models (`repvit_models`).
        If the value is not in the list, a `ValueError` is raised.

        Returns:
            self: The current instance of the `configuration` class.

        Raises:
            ValueError: If the `repvit_model` attribute is not a valid value.
        """

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
        # if self.batch_size % self.cached_batch_size != 0:
        #     raise ValueError(
        #         f"cached_batch_size must be a a multiple of batch_size,
        #          got {self.batch_size=}, but got {self.cached_batch_size=}"
        #     )
        return self

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"DataloaderConfig(batch_size={self.batch_size}, repvit_model={self.repvit_model})"


class TransformerConfig(BaseModel):
    """
    Configuration class for transformer settings.
    Attributes:
        d_model (int, optional): The dimensionality of the model. Defaults to 384.
        n_heads (int, optional): The number of attention heads. Defaults to 2.
        n_layers (int, optional): The number of transformer layers. Defaults to 1.
        d_ff (int, optional): The dimensionality of the feed-forward layer. Defaults to 1024.
    """

    d_model: int = Field(default=384)
    n_heads: int = Field(default=2)
    n_layers: int = Field(default=1)
    d_ff: int = Field(default=1024)
    dropout: float = Field(default=0.1)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            f"TransformerConfig(d_model={self.d_model}, n_heads={self.n_heads}, "
            f"n_layers={self.n_layers}, d_ff={self.d_ff}), dropout={self.dropout}"
        )


class TrainConfig(BaseModel):
    """
    Configuration class for training.
    Attributes:
        learning_rate (float): The learning rate for the training process. Default is 0.001.
        epochs (int): The number of epochs for the training process. Default is 1.
        log_dir (str): The directory to store the training logs.
        early_stop_counter (int): The counter for early stopping. Default is 10.
        resume_training (bool): Flag indicating whether to resume training from a checkpoint. Default is True.
        train_size (float): The proportion of the dataset to use for training. Default is 0.6.
        val_size (float): The proportion of the dataset to use for validation. Default is 0.3.
        test_size (float): The proportion of the dataset to use for testing. Default is 0.1.
    """

    learning_rate: float = Field(default=0.001)
    epochs: int = Field(default=1)
    tmp_dir: str = Field(default="./tmp")
    log_dir: str
    early_stop_counter: int = Field(default=10)
    resume_training: bool = Field(default=True)
    train_size: float = Field(default=0.6)
    val_size: float = Field(default=0.3)
    test_size: float = Field(default=0.1)

    @model_validator(mode="after")
    def check_values(self):
        """
        Custom model validator to check if the sum of train_size, val_size, and test_size is equal to 1.0.
        Raises:
            ValueError: If the sum of train_size, val_size, and test_size is not equal to 1.0.
        Returns:
            TrainConfig: The updated TrainConfig object.
        """
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
            f"val_size={self.val_size}, test_size={self.test_size})"
        )


class ExperimentConfig(BaseModel):
    """
    Configuration class for Megatron_DeepFake experiment.

    Attributes:
        dataset (DatasetConfig): Configuration for the dataset.
        dataloader (DataloaderConfig): Configuration for the dataloader.
        transformer (TransformerConfig): Configuration for the transformer.
        train (TrainConfig): Configuration for the training.
        seed (int): Random seed value. Default is 42.
    .
    """

    dataset: DatasetConfig
    dataloader: DataloaderConfig = Field(default_factory=DataloaderConfig)
    transformer: TransformerConfig = Field(default_factory=TransformerConfig)
    train: TrainConfig
    seed: int = Field(default=42)
    techniques: Optional[list[str]] = Field(default=None)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            f"Config(\n  dataset={self.dataset},\n  dataloader={self.dataloader},\n  "
            f"transformer={self.transformer},\n  train={self.train},\n  seed={self.seed})"
        )
