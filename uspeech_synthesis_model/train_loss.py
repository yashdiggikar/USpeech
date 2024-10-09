import torch
import torch.nn as nn

class SpectrumLoss(nn.Module):
    def __init__(self):
        super(SpectrumLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.epsilon = 1e-8

    def forward(self, output, target):

        mse = self.mse_loss(output, target)
        temporal = self.temporal_difference(output, target)

        total_loss = mse + temporal

        result ={
            'mse': mse,
            'temporal': temporal,
            'total_loss': total_loss
        }
        return result

    def temporal_difference(self, output, target):
        time_diff_output = torch.mean(torch.abs(output[:, :, 1:, :] - output[:, :, :-1, :]))
        time_diff_target = torch.mean(torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :]))
        loss = self.mse_loss(time_diff_output, time_diff_target)
        return loss