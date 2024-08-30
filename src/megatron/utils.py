"""Utilities functions"""

# TODO: Jose, controlla che le utiliti function lavorino come ci aspettiamo
import os
from pathlib import Path
from typing import Tuple, Optional
import torch
from torch import nn


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    save_path: os.PathLike,
):
    """
    Save the checkpoint of the model during training.
    """
    model_name = str(type(model).__name__)
    checkpoint_path = Path(save_path) / "models" / model_name

    print(f"Saving checkpoint to: {checkpoint_path}")

    if not checkpoint_path.exists():
        os.makedirs(checkpoint_path)
        print(f"Created directory: {checkpoint_path}")

    checkpoint = checkpoint_path / f"{model_name}_epoch_{epoch}.pth"
    
    print(f"Final checkpoint path: {checkpoint}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint,
    )
    print("Checkpoint saved successfully.")


def load_model(model: nn.Module, model_path: os.PathLike):
    """Loads into `model` the dictionary found at `model_path`"""
    model_path = Path(model_path)
    model.load_state_dict(torch.load(Path(model_path)))


def load_checkpoint(
    model, checkpoint_path
) -> Tuple[nn.Module, int, torch.optim.Optimizer]:
    """
    Loads a checkpoint for a given model.

    Args:
        model (nn.Module): The model to load the checkpoint for.
        save_path (str): The path where the checkpoint is saved.

    Returns:
        tuple: A tuple containing the loaded model, epoch number, optimizer, and early stop counter.
    """

    epoch = int(checkpoint_path.split("_epoch_")[1].split(".pth")[0])
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Checkpoint loaded: {checkpoint}, starting from epoch: {epoch}")
    return model, epoch, optimizer


def get_model_path(model_root: os.PathLike) -> Optional[os.PathLike]:
    """Returns the model path if exists."""
    model_root = Path(model_root)
    for file in model_root.glob("*.pth"):
        if "epoch" not in file.stem:
            return file
    return None


def get_latest_checkpoint_path(model_root: os.PathLike) -> Optional[os.PathLike]:
    """Returns the latest checkpoint path if exists."""
    model_root = Path(model_root)
    filtered = list(filter(lambda file: "epoch" in file.stem, model_root.glob("*.pth")))
    if not filtered:
        return None

    return sorted(
        filtered,
        key=lambda file: int(file.stem.split("_epoch_")[1].split(".pth")[0]),
    )[-1]


def get_latest_epoch(save_path: os.PathLike) -> int:
    """
    Get the latest epoch number from the checkpoint directory.

    Args:
        save_path (os.PathLike): The path to the checkpoint directory.

    Returns:
        int: The latest epoch number.
    """
    checkpoint_path = Path(save_path) / "models"
    checkpoints_epoch = []
    print(f"Checking directory: {os.path.exists(checkpoint_path)=}")
    for c in checkpoint_path.glob("*.pth"):
        print(f"Checking file: {c.name}")
        try:
            checkpoints_epoch.append(int(c.stem.split("_")[-1]))
        except ValueError:
            print(f"Skipping non-epoch file: {c.name}")

    if checkpoints_epoch:
        latest_epoch = max(checkpoints_epoch)
        print("Latest epoch found: ", latest_epoch)
        return latest_epoch

    print("No checkpoints found")
    return 0


def save_model(model: nn.Module, save_path: os.PathLike):
    """Saves the model state dict.

    Args:
        model (nn.Module): The model to be saved.
        save_path (os.PathLike): The path to save the model.

    """
    model_name = str(type(model).__name__)
    model_path = Path(save_path) / "models" / model_name / f"{model_name}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")
