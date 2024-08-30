"""Utility module for operation on model state."""

from pathlib import Path
from typing import Optional
import torch
from torch import nn
from torch.optim.optimizer import Optimizer


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    optimizer: Optimizer,
    save_path: Path,
) -> None:
    """
    Save the model checkpoint  along with the optimizer state.

    Args:
        model (nn.Module): The model to be saved.
        epoch (int): The current epoch number.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        save_path (Path): The path where the checkpoint will be saved.
    Returns:
        None
    """
    checkpoint_path = save_path / "models"
    model_name = model.__class__.__name__

    checkpoint_path.mkdir(parents=True, exist_ok=True)

    checkpoint = checkpoint_path / f"{model_name}_epoch_{epoch}.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint,
    )


def load_checkpoint(model, optimizer, checkpoint_path) -> None:
    """
    Loads the model and optimizer states from a checkpoint file.
    Works **INPLACE**, so nothing gets moved around memory for no reason.
    Args:
        model (nn.Module): The model to load the state_dict into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state_dict into.
        checkpoint_path (str): The path to the checkpoint file.
    Returns:
        None
    """
    checkpoint = torch.load(checkpoint_path, weights_only=True)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def save_model(model: nn.Module, save_path: Path) -> None:
    """
    Save the state dictionary of a PyTorch model to a specified path.

    Args:
        model (nn.Module): The PyTorch model to be saved.
        save_path (Path): The path where the model state dictionary will be saved.

    Returns:
        None
    """
    model_path = save_path / "models" / f"{model.__class__.__name__}.pth"
    torch.save(model.state_dict(), model_path)


def load_model(model: nn.Module, save_path: Path) -> None:
    """
    Loads the state dictionary of a model from the specified save path.
    Works **INPLACE**, so nothing gets moved around memory for no reason.

    Args:
        model (nn.Module): The model to load the state dictionary into.
        save_path (Path): The path to the directory containing the saved model.

    Returns:
        None
    """
    model_path = save_path / "models" / f"{model.__class__.__name__}.pth"
    model.load_state_dict(torch.load(model_path, weights_only=True))


def get_latest_checkpoint(model_root: Path) -> Optional[Path]:
    """
    Returns the path of the latest checkpoint file in the given model root directory.

    Args:
        model_root (Path): The root directory where the checkpoint files are located.

    Returns:
        Optional[Path]: The path of the latest checkpoint file, or None if no checkpoint files are found.
    """
    checkpoints = sorted(
        (f for f in model_root.glob("*.pth") if "epoch" in f.stem),
        key=lambda x: int(x.stem.split("_epoch_")[1]),
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None
