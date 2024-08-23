"""Definition of one the main transformer, Transformer One"""

import torch
from torch import nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """
    CrossAttention module that performs cross-attention between image and depth embeddings.
    Args:
        d_image (int): The dimension of the image embeddings.
        d_depth (int): The dimension of the depth embeddings.
        d_attn (int): The dimension of the attention.
    Attributes:
        d_image (int): The dimension of the image embeddings.
        d_depth (int): The dimension of the depth embeddings.
        d_attn (int): The dimension of the attention.
        W_Q_image (nn.Linear): Linear layer for image query projection.
        W_K_depth (nn.Linear): Linear layer for depth key projection.
        W_V_depth (nn.Linear): Linear layer for depth value projection.
        W_Q_depth (nn.Linear): Linear layer for depth query projection.
        W_K_image (nn.Linear): Linear layer for image key projection.
        W_V_image (nn.Linear): Linear layer for image value projection.
        W_final (nn.Linear): Linear layer for final output projection.
    Methods:
        forward(image_embeddings, depth_embeddings):
            Performs forward pass of the cross-attention module.
    """

    def __init__(self, d_image: int, d_depth: int, d_attn: int):
        super().__init__()
        self.d_image = d_image
        self.d_depth = d_depth
        self.d_attn = d_attn

        self.W_Q_image = nn.Linear(d_image, d_attn)
        self.W_K_depth = nn.Linear(d_depth, d_attn)
        self.W_V_depth = nn.Linear(d_depth, d_attn)

        self.W_Q_depth = nn.Linear(d_depth, d_attn)
        self.W_K_image = nn.Linear(d_image, d_attn)
        self.W_V_image = nn.Linear(d_image, d_attn)

        self.W_final = nn.Linear(2 * d_attn, d_attn)

    def forward(
        self, image_embeddings: torch.Tensor, depth_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs forward pass of the cross-attention module.
        Args:
            image_embeddings (torch.Tensor): The image embeddings.
            depth_embeddings (torch.Tensor): The depth embeddings.
        Returns:
            final_output (torch.Tensor): The final output of the cross-attention module.
        """
        Q_image = self.W_Q_image(image_embeddings)
        K_depth = self.W_K_depth(depth_embeddings)
        V_depth = self.W_V_depth(depth_embeddings)

        attn_scores_image_to_depth = torch.bmm(Q_image, K_depth.transpose(1, 2)) / (
            self.d_attn**0.5
        )  # (batch_size, seq_len, seq_len)
        attn_weights_image_to_depth = F.softmax(
            attn_scores_image_to_depth, dim=-1
        )  # (batch_size, seq_len, seq_len)
        output_image_to_depth = torch.bmm(
            attn_weights_image_to_depth, V_depth
        )  # (batch_size, seq_len, d_attn)

        Q_depth = self.W_Q_depth(depth_embeddings)
        K_image = self.W_K_image(image_embeddings)
        V_image = self.W_V_image(image_embeddings)

        attn_scores_depth_to_image = torch.bmm(Q_depth, K_image.transpose(1, 2)) / (
            self.d_attn**0.5
        )  # (batch_size, seq_len, seq_len)
        attn_weights_depth_to_image = F.softmax(
            attn_scores_depth_to_image, dim=-1
        )  # (batch_size, seq_len, seq_len)
        output_depth_to_image = torch.bmm(
            attn_weights_depth_to_image, V_image
        )  # (batch_size, seq_len, d_attn)

        concat_output = torch.cat(
            [output_image_to_depth, output_depth_to_image], dim=-1
        )  # (batch_size, seq_len, 2 * d_attn)

        final_output = self.W_final(concat_output)  # (batch_size, seq_len, d_attn)

        return final_output


class TransformerEncoder(nn.Module):
    """
    TransformerEncoder is a module that applies a stack of TransformerEncoderLayers to the input features.
    Args:
        d_model (int): The number of expected features in the input.
        n_heads (int): The number of attention heads.
        n_layers (int): The number of TransformerEncoderLayers in the stack.
        d_ff (int): The dimension of the feedforward network.
    Attributes:
        layers (nn.ModuleList) : TODO
        cros_attn (CrossAttention) : TODO
    Methods:
        forward(x): Performs a forward pass of the Transformer Encoder.
    """

    def __init__(
        self,
        d_model,
        n_heads,
        n_layers,
        d_ff,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.cross_attn = CrossAttention(d_model, d_model, d_model)

    def forward(
        self, rgb_features: torch.Tensor, depth_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs forward pass through the network.
        Args:
            rgb_features (torch.Tensor): Input RGB features.
            depth_features (torch.Tensor): Input depth features.
        Returns:
            torch.Tensor: Output of the forward pass.
        """
        for layer in self.layers:
            rgb_embd = layer(rgb_features)
            depth_embd = layer(depth_features)

        output = self.cross_attn(rgb_embd, depth_embd)

        return output


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the Transformer Encoder.
    Args:
        d_model (int): The number of expected features in the input.
        n_heads (int): The number of heads in the multihead attention mechanism.
        d_ff (int): The dimension of the feedforward network.
        dropout (float, optional): The dropout probability. Default is 0.1.
    Attributes:
        self_attn (nn.MultiheadAttention): The multihead self-attention module.
        linear1 (nn.Linear): The first linear transformation module.
        linear2 (nn.Linear): The second linear transformation module.
        norm1 (nn.LayerNorm): The first layer normalization module.
        norm2 (nn.LayerNorm): The second layer normalization module.
        dropout (nn.Dropout): The dropout module.
    Methods:
        forward(x): Performs a forward pass of the Transformer Encoder layer.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x2 = self.norm1(x)
        x = x + self.self_attn(x2, x2, x2)[0]  # skip connection
        x2 = self.norm2(x)
        x = x + self.dropout(F.relu(self.linear1(x2)))  # skip connection with dropout
        x = self.linear2(x)
        return x


class FeatureProjector(nn.Module):
    """
    A module that projects input features to a different dimension.
    Args:
        d_input (int): The input dimension of the features.
        d_output (int): The output dimension of the projected features.
    Attributes:
        projector (nn.Linear): The linear projection layer.
    Methods:
        forward(features): Projects the input features to the output dimension.
    """

    def __init__(self, d_input: int, d_output: int):
        super().__init__()
        self.projector = nn.Linear(d_input, d_output)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Applies the projection operation on the input features.

        Args:
            features (torch.Tensor): The input features to be projected.

        Returns:
            torch.Tensor: The projected features.

        """
        return self.projector(features)


class TransformerFakeDetector(nn.Module):
    """
    A class representing a Transformer-based fake detector.
    Args:
        d_input_features (int): The number of input features.
        d_model (int): The dimensionality of the model.
        n_heads (int): The number of attention heads.
        n_layers (int): The number of transformer layers.
        d_ff (int): The dimensionality of the feed-forward layer.
        num_classes (int): The number of output classes.
    Attributes:
        projector (FeatureProjector): The feature projector module.
        encoder (TransformerEncoder): The transformer encoder module.
        classifier (nn.Linear): The linear classifier layer.
    Methods:
        forward(rgb_features, depth_features): Performs a forward pass through the network.
    """

    def __init__(self, d_input_features, d_model, n_heads, n_layers, d_ff, num_classes):
        super().__init__()
        self.projector = FeatureProjector(d_input_features, d_model)
        self.encoder = TransformerEncoder(d_model, n_heads, n_layers, d_ff)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, rgb_features: torch.Tensor, depth_features: torch.Tensor):
        """
        Performs a forward pass through the network.
        Args:
            rgb_features (torch.Tensor): The RGB input features.
            depth_features (torch.Tensor): The depth input features.
        Returns:
            torch.Tensor: The softmax probabilities of the output classes.
        """
        rgb_projected = self.projector(rgb_features)
        depth_projected = self.projector(depth_features)

        output = self.encoder(rgb_projected, depth_projected)
        logits = self.classifier(output)
        return F.softmax(logits, dim=-1)
