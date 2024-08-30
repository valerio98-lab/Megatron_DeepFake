"""Definition of one the main transformer, Transformer One"""

import torch
from torch import nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
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

        # Initialize weights
        self.init_weights()

        # Layer normalization for stability before softmax
        self.norm_image = nn.LayerNorm(d_attn)
        self.norm_depth = nn.LayerNorm(d_attn)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self, image_embeddings: torch.Tensor, depth_embeddings: torch.Tensor
    ) -> torch.Tensor:
        Q_image = self.W_Q_image(image_embeddings)
        K_depth = self.W_K_depth(depth_embeddings)
        V_depth = self.W_V_depth(depth_embeddings)

        Q_image = self.norm_image(Q_image)
        K_depth = self.norm_depth(K_depth)

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

        # Apply normalization before softmax
        Q_depth = self.norm_depth(Q_depth)
        K_image = self.norm_image(K_image)

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
    def __init__(
        self,
        d_model,
        n_heads,
        n_layers,
        d_ff,
        dropout=0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.cross_attn = CrossAttention(d_model, d_model, d_model)
        self.norm_rgb = nn.LayerNorm(d_model)
        self.norm_depth = nn.LayerNorm(d_model)

    def forward(
        self, rgb_features: torch.Tensor, depth_features: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            rgb_features = layer(rgb_features)
            depth_features = layer(depth_features)

        rgb_features = self.norm_rgb(rgb_features)
        depth_features = self.norm_depth(depth_features)

        output = self.cross_attn(rgb_features, depth_features)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm1 = self.norm1(x)
        x_attn = self.self_attn(x_norm1, x_norm1, x_norm1)[0]
        x = x + x_attn  # skip connection
        x_norm2 = self.norm2(x)
        x_f1 = self.dropout(F.relu(self.linear1(x_norm2)))
        x = x + self.linear2(x_f1)  # skip connection with dropout
        return x


class FeatureProjector(nn.Module):
    def __init__(
        self, d_input: int, d_output: int, add_nonlinearity=False, dropout_rate=0.0
    ):
        super().__init__()
        self.projector = nn.Linear(d_input, d_output)
        self.add_nonlinearity = add_nonlinearity
        self.activation = nn.ReLU() if add_nonlinearity else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.projector.weight)
        nn.init.constant_(self.projector.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.projector(features)
        features = self.activation(features)
        features = self.dropout(features)
        return features


class TransformerFakeDetector(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        n_layers,
        d_ff,
        num_classes,
        dropout=0.1,
        d_input_features=None,
        projector_bool=False,
    ):
        super().__init__()
        self.projector_bool = projector_bool
        if d_input_features is not None:
            self.projector = FeatureProjector(d_input_features, d_model)
        self.encoder = TransformerEncoder(d_model, n_heads, n_layers, d_ff, dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        self.projector_bool = projector_bool
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, rgb_batch, depth_batch):

        if self.projector_bool:
            rgb_batch = self.projector(rgb_batch)
            depth_batch = self.projector(depth_batch)

        output = self.encoder(rgb_batch, depth_batch)
        output = self.pool(output.transpose(1, 2)).squeeze(-1)
        # output = self.dropout(output)
        logits = self.classifier(output)

        return logits
