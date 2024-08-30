from pathlib import Path
from typing import Optional
import torch
from torch import nn


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    save_path: Path,
) -> None:
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
    checkpoint = torch.load(checkpoint_path, weights_only=True)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def save_model(model: nn.Module, save_path: Path) -> None:
    model_path = save_path / "models" / f"{model.__class__.__name__}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")


def load_model(model: nn.Module, model_path: Path) -> None:
    model.load_state_dict(torch.load(model_path))


def get_latest_checkpoint(model_root: Path) -> Optional[Path]:
    checkpoints = sorted(
        (f for f in model_root.glob("*.pth") if "epoch" in f.stem),
        key=lambda x: int(x.stem.split("_epoch_")[1]),
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None
