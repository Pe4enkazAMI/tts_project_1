import numpy as np
import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, q, k, v, mask=None):
        attn = q @ k.transpose(-1, -2)
        attn = attn / self.temperature
        if mask is not None:
            attn.masked_fill(mask, -torch.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = attn @ v
        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head # num head
        self.d_k = d_k # litteraly head_dim
        self.d_v = d_v # literally head_dim 
        self.d_model = d_model # emb dim

        # new____________________________________________
        self.qkv = nn.Linear(self.d_model, 3*self.d_model)
        self.out = nn.Linear(self.d_model, self.d_model)
        #________________________________________________

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5) 
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.xavier_normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.xavier_normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_v))) 
        
    def forward(self, x, mask=None):
        residual = x 
        bs, seq_len, _  = x.shape
        x = self.layer_norm(x)
        qkv = self.qkv(x)
        qkv = qkv.reshape(bs, seq_len, self.n_head, 3 * self.d_v)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)   # b x n x .. x ..
            
        output, attn = self.attention(q, k, v, mask=mask)
        output = output.permute(0, 2, 1, 3)
        output = output.reshape(bs, seq_len, self.d_model)
        output = self.dropout(self.out(output))
        output = output + residual

        return output, attn