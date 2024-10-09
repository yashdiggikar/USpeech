import torch.nn as nn

class UnetLoss(nn.Module):
    def __init__(self):

        super(UnetLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, gt):
        loss = self.loss(pred, gt)
        result = {
            'loss': loss
        }
        return result