import torch.nn as nn
import torch

class FSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, mel_pred, duration_pred, pitch_pred, energy_pred, 
                      gt_mel, gt_duration, gt_pitch, gt_energy, **kwargs):
        loss_mel = self.mse(mel_pred, gt_mel)
        loss_dur = self.mse(
            duration_pred,
            torch.log1p(gt_duration.to(torch.float))
        )
        loss_energy = self.mse(
            energy_pred,
            torch.log1p(gt_energy.to(torch.float))
        )
        loss_pitch = self.mse(
            pitch_pred,
            torch.log1p(gt_pitch.to(torch.float))
        )
        return loss_mel, loss_dur, loss_energy, loss_pitch