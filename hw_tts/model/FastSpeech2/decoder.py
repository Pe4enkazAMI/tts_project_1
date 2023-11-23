import torch.nn as nn
from .model_utils import get_attn_key_pad_mask, get_non_pad_mask
from .FFT import FFTBlock

class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, max_seq_len, 
        decoder_n_layer,
        decoder_dim,
        decoder_head,
        decoder_conv1d_filter_size,
        fft_conv1d_kernel,
        fft_conv1d_padding,
        PAD,
        dropout):

        super().__init__()

        len_max_seq=max_seq_len
        n_position = len_max_seq + 1
        n_layers = decoder_n_layer

        self.position_enc = nn.Embedding(
            n_position,
            decoder_dim,
            padding_idx=PAD,
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            decoder_dim,
            decoder_conv1d_filter_size,
            decoder_head,
            decoder_dim // decoder_head,
            decoder_dim // decoder_head,
            fft_conv1d_kernel,
            fft_conv1d_padding,
            dropout=dropout
        ) for _ in range(n_layers)])

        self.PAD = PAD

    def forward(self, enc_seq, enc_pos, return_attns=False):

        dec_slf_attn_list = []

        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos, PAD=self.PAD)
        non_pad_mask = get_non_pad_mask(enc_pos, self.PAD)

        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output
