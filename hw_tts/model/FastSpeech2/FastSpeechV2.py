import numpy as np
import torch
from .LenReg import LengthRegulator

from .predictors import VariancePredictor
import torch.nn as nn
from .model_utils import get_mask_from_lengths
from .encoder import Encoder
from .decoder import Decoder


class FastSpeechModel(nn.Module):
    """ FastSpeech """

    def __init__(self, 
                 max_seq_len,
                 encoder_n_layer,
                 decoder_n_layer,
                 vocab_size,
                 encoder_dim,
                 encoder_head,
                 encoder_filter_size,
                 decoder_dim,
                 decoder_head,
                 decoder_filter_size,
                 fft_kernel=[9, 1],
                 fft_padding=[4, 0],
                 duration_predictor_filter_size=256,
                 duration_predictor_kernel_size=3,
                 pitch_predictor_filter_size=256,
                 pitch_predictor_kernel_size=3,
                 energy_predictor_filter_size=256,
                 energy_predictor_kernel_size=3,
                 min_pitch=59.913448819015024,
                 max_pitch=887.2688230720693,
                 min_energy=15.023643,
                 max_energy=91.4197,
                 num_bins=256,
                 num_mels=80,
                 PAD=0,
                 dropout=0.1
        ):
        super().__init__()

        self.encoder = Encoder(max_seq_len, 
                               encoder_n_layer,
                               vocab_size,
                               encoder_dim,
                               encoder_head,
                               encoder_filter_size,
                               fft_kernel,
                               fft_padding,
                               PAD,
                               dropout
        )
        self.length_regulator = LengthRegulator(encoder_dim, 
                                                duration_predictor_filter_size,
                                                duration_predictor_kernel_size,
                                                dropout)
        self.decoder = Decoder(max_seq_len, 
                               decoder_n_layer,
                               decoder_dim,
                               decoder_head,
                               decoder_filter_size,
                               fft_kernel,
                               fft_padding,
                               PAD,
                               dropout)

        self.mel_linear = nn.Linear(decoder_dim, num_mels)

        pitch_space = torch.linspace(np.log(min_pitch + 1),
                                     np.log(max_pitch + 2), 
                                     num_bins)
        self.register_buffer('pitch_space', pitch_space)

        self.pitch_emb = nn.Embedding(num_bins, encoder_dim)
        self.pitch_predictor = VariancePredictor(encoder_dim,
                                                 pitch_predictor_filter_size,
                                                 pitch_predictor_kernel_size,
                                                 dropout)

        energy_space = torch.linspace(np.log(min_energy + 1),
                                      np.log(max_energy + 2),
                                      num_bins)
        self.register_buffer('energy_space', energy_space)

        self.energy_emb = nn.Embedding(num_bins, encoder_dim)
        self.energy_predictor = VariancePredictor(encoder_dim,
                                                  energy_predictor_filter_size,
                                                  energy_predictor_kernel_size,
                                                  dropout)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)
    
    def get_entity(self, x, predictor, space, entity_emb, target=None, scale=1.0):
        entity_predictor_output = predictor(x)

        max_val = entity_emb.num_embeddings
        if target is not None:
            buckets = torch.bucketize(torch.log1p(target), space)
        else:
            estimated_entity = torch.expm1(entity_predictor_output)
            estimated_entity = estimated_entity * scale
            buckets = torch.bucketize(torch.log1p(estimated_entity), space)
        
        buckets = torch.clip(buckets, min=0, max=max_val - 1)
        emb = entity_emb(buckets)
        return emb, entity_predictor_output

    def forward(self, src_seq, src_pos, mel_pos=None,
                mel_max_length=None, gt_duration=None, 
                gt_pitch=None, gt_energy=None,
                alpha=1.0, beta=1.0, gamma=1.0,
                **kwargs):
        x, non_pad_mask = self.encoder(src_seq, src_pos)
        if self.training:
            output, duration_predictor_output = self.length_regulator(x, 
                                                                      alpha, 
                                                                      gt_duration, 
                                                                      mel_max_length)
            
            pitch_emb, pitch_predictor_output = self.get_entity(x=output,
                                                                target=gt_pitch,
                                                                predictor=self.pitch_predictor,
                                                                space=self.pitch_space,
                                                                entity_emb=self.pitch_emb,
                                                                scale=beta)
            
            energy_emb, energy_predictor_output = self.get_entity(x=output,
                                                                  target=gt_energy,
                                                                  predictor=self.energy_predictor,
                                                                  space=self.energy_space,
                                                                  entity_emb=self.energy_emb,
                                                                  scale=gamma)
            
            output = self.decoder(output + pitch_emb + energy_emb, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            output = self.mel_linear(output)
            

            return {"mel_pred": output, 
                    "duration_pred": duration_predictor_output,
                    "pitch_pred": pitch_predictor_output,
                    "energy_pred": energy_predictor_output}
        else:
            output, mel_pos = self.length_regulator(x, alpha)
            pitch_emb, _ = self.get_entity(x=output,
                                           predictor=self.pitch_predictor,
                                           space=self.pitch_space,
                                           entity_emb=self.pitch_emb,
                                           scale=beta)
            energy_emb, _ = self.get_entity(x=output,
                                            predictor=self.energy_predictor,
                                            space=self.energy_space,
                                            entity_emb=self.energy_emb,
                                            scale=gamma)
            output = self.decoder(output + pitch_emb + energy_emb, mel_pos)
            output = self.mel_linear(output)
            return {"mel_output": output}
        

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)