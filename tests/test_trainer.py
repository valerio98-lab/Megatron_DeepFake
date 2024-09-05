"""Test training"""

import random
import numpy as np
import torch

from megatron.configuration import (
    DataloaderConfig,
    DatasetConfig,
    ExperimentConfig,
    TrainConfig,
    TransformerConfig,
)
from megatron.trainer import Trainer

if __name__ == "__main__":

    experiment = ExperimentConfig(
        dataset=DatasetConfig(
            video_path=r"H:\My Drive\Megatron_DeepFake\dataset_processed",
            num_frames=5,
            random_initial_frame=False,
            depth_anything_size="Small",
            num_video=5,
        ),
        dataloader=DataloaderConfig(
            batch_size=8,
        ),
        transformer=TransformerConfig(
            d_model=384,
            n_heads=2,
            n_layers=1,
            d_ff=1024,
        ),
        train=TrainConfig(
            learning_rate=0.001,
            epochs=3,
            tmp_dir="./tmp",
            log_dir="./data/exp1",
            resume_training=False,
            train_size=0.5,
            val_size=0.4,
        ),
        seed=42,
    )
    torch.backends.cudnn.deterministic = True
    random.seed(experiment.seed)
    np.random.seed(experiment.seed)
    torch.manual_seed(experiment.seed)
    trainer = Trainer(experiment)
    trainer.optimized_train_and_validate()
