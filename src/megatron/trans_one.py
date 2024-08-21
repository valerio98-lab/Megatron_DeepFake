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
    

class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, activation):
        super(Transformer, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
        
    def forward(self, src, tgt):
        output = self.transformer(src, tgt)
        return output



