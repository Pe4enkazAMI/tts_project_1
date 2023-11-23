import torch.nn as nn
from .model_utils import get_attn_key_pad_mask, get_non_pad_mask
from .FFT import FFTBlock

class Encoder(nn.Module):
    def __init__(self, 
        max_seq_len, 
        encoder_n_layer,
        vocab_size,
        encoder_dim,
        encoder_head,
        encoder_filter_size,
        fft_kernel,
        fft_padding,
        PAD,
        dropout
        ):
        super().__init__()
        
        len_max_seq= max_seq_len
        n_position = len_max_seq + 1
        n_layers = encoder_n_layer

        self.src_word_emb = nn.Embedding(
            vocab_size,
            encoder_dim,
            padding_idx=PAD
        )

        self.position_enc = nn.Embedding(
            n_position,
            encoder_dim,
            padding_idx=PAD
        )

        self.layer_stack = nn.ModuleList([
            FFTBlock(encoder_dim, 
                     encoder_filter_size,
                     encoder_head,
                     encoder_dim // encoder_head,
                     encoder_dim // encoder_head,
                     fft_kernel,
                     fft_padding,
                     dropout=dropout) for _ in range(n_layers)])
        self.PAD = PAD

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq, PAD=self.PAD)
        non_pad_mask = get_non_pad_mask(src_seq, self.PAD)
        
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output, non_pad_mask