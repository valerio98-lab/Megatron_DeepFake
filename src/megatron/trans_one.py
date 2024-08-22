import torch 
import torch.nn as nn
import torch.nn.functional as F 

class CrossAttention(nn.Module):
    def __init__(self, d_image, d_depth, d_attn):
        super(CrossAttention, self).__init__()
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

    def forward(self, image_embeddings, depth_embeddings):
        Q_image = self.W_Q_image(image_embeddings)
        K_depth = self.W_K_depth(depth_embeddings)
        V_depth = self.W_V_depth(depth_embeddings)

        attn_scores_image_to_depth = torch.bmm(Q_image, K_depth.transpose(1, 2)) / (self.d_attn ** 0.5)  # (batch_size, seq_len, seq_len)
        attn_weights_image_to_depth = F.softmax(attn_scores_image_to_depth, dim=-1)  # (batch_size, seq_len, seq_len)
        output_image_to_depth = torch.bmm(attn_weights_image_to_depth, V_depth)  # (batch_size, seq_len, d_attn)
        
        Q_depth = self.W_Q_depth(depth_embeddings)
        K_image = self.W_K_image(image_embeddings)
        V_image = self.W_V_image(image_embeddings)

        attn_scores_depth_to_image = torch.bmm(Q_depth, K_image.transpose(1, 2)) / (self.d_attn ** 0.5)  # (batch_size, seq_len, seq_len)
        attn_weights_depth_to_image = F.softmax(attn_scores_depth_to_image, dim=-1)  # (batch_size, seq_len, seq_len)
        output_depth_to_image = torch.bmm(attn_weights_depth_to_image, V_image)  # (batch_size, seq_len, d_attn)


        concat_output = torch.cat([output_image_to_depth, output_depth_to_image], dim=-1)  # (batch_size, seq_len, 2 * d_attn)

        final_output = self.W_final(concat_output)  # (batch_size, seq_len, d_attn)

        return final_output
    

class TransformerEncoder(nn.Module):
    def __init__(
            self, 
            d_model,
            n_heads,
            n_layers,
            d_ff,
    ): 
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.cross_attn = CrossAttention(d_model, d_model, d_model)
    
    def forward(self, rgb_features, depth_features):
        for layer in self.layers: 
            rgb_embd = layer(rgb_features)
            depth_embd = layer(depth_features)
        
        output = self.cross_attn(rgb_embd, depth_embd)

        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(
            self, 
            d_model, 
            n_heads,
            d_ff,
            dropout=0.1
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x2 = self.norm1(x)
        x = x + self.self_attn(x2, x2, x2)[0] #skip connection
        x2 = self.norm2(x)
        x = x + self.dropout(F.relu(self.linear1(x2))) #skip connection with dropout
        x = self.linear2(x)
        return x

class FeatureProjector(nn.Module):
    def __init__(self, d_input, d_output): 
        super(FeatureProjector, self).__init__()
        self.projector = nn.Linear(d_input, d_output)
    
    def forward(self, features):
        return self.projector(features)


class TransformerFakeDetector(nn.Module):
    def __init__(self, d_input_features, d_model, n_heads, n_layers, d_ff, num_classes):
        super(TransformerFakeDetector, self).__init__()
        self.projector = FeatureProjector(d_input_features, d_model)
        self.encoder = TransformerEncoder(d_model, n_heads, n_layers, d_ff)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, rgb_features, depth_features):
        rgb_projected = self.projector(rgb_features)
        depth_projected = self.projector(depth_features)

        output = self.encoder(rgb_projected, depth_projected)
        logits = self.classifier(output)
        return F.softmax(logits, dim=-1)
