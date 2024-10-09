import torch
import torch.nn as nn


class down_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(down_conv, self).__init__()
        self.downconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True))

    def forward(self, downconv_input):
        return self.downconv(downconv_input)

# pixel shuffle
class up_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(True),
            nn.PixelShuffle(2))

        self.upconv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))

    def forward(self, upconv_input1, upconv_input2):
        upconv_input1 = self.up(upconv_input1)
        output = torch.cat([upconv_input1, upconv_input2], dim=1)
        output = self.upconv(output)
        return output


class inout_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(inout_conv, self).__init__()
        self.inoutconv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))

    def forward(self, inout_input):
        return self.inoutconv(inout_input)

class in_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(in_conv, self).__init__()
        self.inconv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=(9,1), padding=(4,0)),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True),
                                nn.Conv2d(out_channels, out_channels, kernel_size=(1,9), padding=(0,4)),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
    def forward(self, input_amp):
        return self.inconv(input_amp)


class out_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(out_conv, self).__init__()
        self.inoutconv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))

    def forward(self, inout_input):
        return self.inoutconv(inout_input)