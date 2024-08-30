import os
import pytest
import torch
import shutil
from torch import nn
from torch.optim import Adam

from megatron.utils import (
    save_checkpoint,
    load_checkpoint,
    load_model,
    get_model_path,
    get_latest_checkpoint_path,
    get_latest_epoch,
    save_model
)

class SimpleModel(nn.Module):
    """Simple neural network model for testing purposes."""

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture(scope="module")
def setup_model_and_dir():
    """Fixture to set up the model, optimizer, and test directory."""
    test_dir = ""
    os.makedirs(test_dir, exist_ok=True)
    model = SimpleModel()
    optimizer = Adam(model.parameters(), lr=0.001)

    yield model, optimizer, test_dir
    shutil.rmtree(test_dir) 


def test_save_and_load_checkpoint(setup_model_and_dir):
    """Test saving and loading a model checkpoint."""
    model, optimizer, test_dir = setup_model_and_dir

    save_checkpoint(model, 0, optimizer, test_dir)
    checkpoint_path = os.path.join(test_dir, "models", "SimpleModel", "SimpleModel_epoch_1.pth")

    assert os.path.exists(checkpoint_path), "Checkpoint file was not created."

    _, loaded_epoch, loaded_optimizer = load_checkpoint(model, checkpoint_path)
    assert loaded_epoch == 0, "Epoch number loaded incorrectly."
    assert isinstance(loaded_optimizer, type(optimizer)), "Optimizer type loaded incorrectly."


def test_save_and_load_model(setup_model_and_dir):
    """Test saving and loading a full model."""
    model, _, test_dir = setup_model_and_dir

    save_model(model, test_dir)
    model_path = os.path.join(test_dir, "models", "SimpleModel", "SimpleModel.pth")

    assert os.path.exists(model_path), "Model file was not created."

    new_model = SimpleModel()
    load_model(new_model, model_path)
    for param_original, param_loaded in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(param_original, param_loaded), "Model parameters were not loaded correctly."


def test_get_latest_checkpoint_path(setup_model_and_dir):
    """Test getting the latest checkpoint path."""
    model, optimizer, test_dir = setup_model_and_dir

    save_checkpoint(model, 0, optimizer, test_dir)
    save_checkpoint(model, 1, optimizer, test_dir)

    latest_checkpoint = get_latest_checkpoint_path(os.path.join(test_dir, "models", "SimpleModel"))
    assert "epoch_1" in latest_checkpoint.name, "Latest checkpoint path is incorrect."



def test_get_latest_epoch(setup_model_and_dir):
    """Test getting the latest epoch number."""
    _, _, test_dir = setup_model_and_dir

    latest_epoch = get_latest_epoch(test_dir)
    assert latest_epoch == 2, "Latest epoch number is incorrect."


def test_get_model_path(setup_model_and_dir):
    """Test getting model path."""
    model, _, test_dir = setup_model_and_dir

    save_model(model, test_dir)
    model_path = get_model_path(os.path.join(test_dir, "models", "SimpleModel"))
    assert model_path is not None, "Model path should exist."


#pytest -v tests/test_utils.py