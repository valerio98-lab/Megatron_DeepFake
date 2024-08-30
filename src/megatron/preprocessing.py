""" Module containing definition of preprocessing models"""

import torch
from torch import nn
import timm  # type: ignore


class PositionalEncoding(nn.Module):
    """
    Positional encoding module for adding positional information to input sequences.

    Attributes:
        pe (torch.Tensor): The positional encoding tensor of shape (max_len, d_model).

    """

    def __init__(self, d_model: int, max_len: int = 50):
        """
        Initializes a positional encoding.

        Args:
            d_model (int): The dimension of the input feature vectors.
            max_len (int, optional): The maximum length of the input sequences. Defaults to 50.
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        forward(x):
            Adds positional encoding to the input sequence.
            Args:
                x (torch.Tensor): The input sequence tensor of shape (seq_len, d_model).
            Returns:
                torch.Tensor: The input sequence tensor with positional encoding added, of shape (seq_len, d_model).
        """
        seq_len, _ = x.size()
        x = x + self.pe[:seq_len, :]
        return x


class RepVit(nn.Module):
    """
    Module for instantiating the RepVit model.
    """

    def __init__(
        self,
        repvit_model: str = "repvit_m0_9.dist_300e_in1k",
    ):
        """
        Initializes an instance of the RepVit model.

        Args:
            repvit_model (str, optional): The name of the repvit model to use. Defaults to "repvit_m0_9.dist_300e_in1k".
        """
        super().__init__()
        self.repvit = timm.create_model(
            repvit_model,
            pretrained=True,
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Calculates the embeddings for the given image tensor using RepVit.

        Args:
            img (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The RepVit-embedding tensor for the input image.
        """
        img = img.float() / 255.0
        return self.repvit.forward_head(
            self.repvit.forward_features(img), pre_logits=True
        )
