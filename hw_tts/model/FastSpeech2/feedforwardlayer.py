import torch.nn as nn 
import torch
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, fft_kernel_size, fft_padding,  dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in, d_hid, kernel_size=fft_kernel_size[0], padding=fft_padding[0])
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=fft_kernel_size[1], padding=fft_padding[1])

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        out = self.layer_norm(x).transpose(1, 2)
        out = self.w_2(F.relu(self.w_1(out))).transpose(1, 2)
        out = self.dropout(out)
        out = out + res
        return out