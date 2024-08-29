from os import PathLike
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# TODO: Jose, rimuovere il parametro frame_threshold, puo' causare problemi allo stack


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
            f"val_size={self.val_size}, test_size={self.test_size})"
        )


class Config(BaseModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    dataloader: DataloaderConfig = Field(default_factory=DataloaderConfig)
    transformer: TransformerConfig = Field(default_factory=TransformerConfig)
    train: TrainConfig
    seed: int = Field(default=42)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            f"Config(\n  dataset={self.dataset},\n  dataloader={self.dataloader},\n  "
            f"transformer={self.transformer},\n  train={self.train},\n  seed={self.seed})"
        )
