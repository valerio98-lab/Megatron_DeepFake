import torch
from torch import nn
from typing import Literal
import timm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
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
        seq_len, _ = x.size()
        x = x + self.pe[:seq_len, :]
        return x


class RepVit(nn.Module):
    def __init__(
        self,
        repvit_model: Literal[
            "repvit_m0_9.dist_300e_in1k",
            "repvit_m2_3.dist_300e_in1k",
            "repvit_m0_9.dist_300e_in1k",
            "repvit_m1_1.dist_300e_in1k",
            "repvit_m2_3.dist_450e_in1k",
            "repvit_m1_5.dist_300e_in1k",
            "repvit_m1.dist_in1k",
        ] = "repvit_m0_9.dist_300e_in1k",
    ):
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
