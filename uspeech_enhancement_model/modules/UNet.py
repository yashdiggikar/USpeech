# adjust the code from Radio2Speech: High Quality Speech Recovery from Radio Frequency Signals
# abs: https://arxiv.org/abs/2206.11066
# code: https://github.com/ZhaoRunning/Radio2Speech

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .UNet_parts import down_conv, up_conv, in_conv, out_conv, inout_conv

def rescale_module(module):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv2d, nn.ConvTranspose2d)):
            sub.weight.data.normal_(0.0, 0.001)
        elif isinstance(sub, nn.BatchNorm2d):
            sub.weight.data.normal_(1.0, 0.02)
            sub.bias.data.fill_(0)

class unet(nn.Module):
    def __init__(self, ngf=32, input_nc=1, output_nc=1, rescale=True): #ngf=64
        super(unet, self).__init__()
        self.rescale = rescale
        self.time_downsample_ratio = 2 ** 3 # This number equals 2^{#encoder_blcoks}

        #initialize layers
        self.inlayer = in_conv(input_nc, ngf)
        self.downlayer1 = down_conv(ngf, ngf*2)
        self.downlayer2 = down_conv(ngf*2, ngf*4)
        self.downlayer3 = down_conv(ngf*4, ngf*8)

        self.uplayer2 = up_conv(ngf*8, ngf*4)
        self.uplayer3 = up_conv(ngf*4, ngf*2)
        self.uplayer4 = up_conv(ngf*2, ngf)
        self.outlayer = inout_conv(ngf, output_nc)

        # weight initialization
        if self.rescale:
            rescale_module(self)

    def forward(self, radio_input):
        radio_input = F.pad(radio_input, pad=(0, 0, 7, 0))
        origin_len = radio_input.size(-1)
        if not origin_len % self.time_downsample_ratio == 0:
            pad_len = int(np.ceil(origin_len / self.time_downsample_ratio)) \
                      * self.time_downsample_ratio - origin_len
            radio_input = F.pad(radio_input, pad=(0, pad_len, 0, 0))

        radio_feature = self.inlayer(radio_input)
        radio_downfeature1 = self.downlayer1(radio_feature)
        radio_downfeature2 = self.downlayer2(radio_downfeature1)
        radio_downfeature3 = self.downlayer3(radio_downfeature2)

        radio_upfeature2 = self.uplayer2(radio_downfeature3, radio_downfeature2)
        radio_upfeature3 = self.uplayer3(radio_upfeature2, radio_downfeature1)
        radio_upfeature4 = self.uplayer4(radio_upfeature3, radio_feature)
        audio_output = self.outlayer(radio_upfeature4)

        if not origin_len % self.time_downsample_ratio == 0:
            audio_output = audio_output[..., : origin_len] # (bs, channels, F, T)

        audio_output = audio_output[..., 7:, :]
        assert audio_output.size(-1) == origin_len

        return audio_output

class unet_upsample(nn.Module):
    def __init__(self, ngf=32, input_nc=1, output_nc=1, rescale=True): #ngf=64
        super(unet_upsample, self).__init__()
        self.rescale = rescale

        #initialize layers
        self.inlayer = inout_conv(input_nc, ngf)
        self.downlayer1 = down_conv(ngf, ngf*2)
        self.downlayer2 = down_conv(ngf*2, ngf*4)
        self.downlayer3 = down_conv(ngf*4, ngf*8)
        self.downlayer4 = down_conv(ngf*8, ngf*16)

        self.uplayer1 = up_conv(ngf*16, ngf*8)
        self.uplayer2 = up_conv(ngf*8, ngf*4)
        self.uplayer3 = up_conv(ngf*4, ngf*2)
        self.uplayer4 = up_conv(ngf*2, ngf)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(ngf, ngf // 2, kernel_size=(4,3), stride=(2,1), padding=(1,1)),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            nn.Conv2d(ngf // 2, ngf // 2, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            nn.Conv2d(ngf // 2, ngf // 2, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True))
        self.outlayer = inout_conv(ngf // 2, output_nc)

        # weight initialization
        if self.rescale:
            rescale_module(self)

    def forward(self, radio_input):
        radio_feature = self.inlayer(radio_input)
        radio_downfeature1 = self.downlayer1(radio_feature)
        radio_downfeature2 = self.downlayer2(radio_downfeature1)
        radio_downfeature3 = self.downlayer3(radio_downfeature2)
        radio_downfeature4 = self.downlayer4(radio_downfeature3)

        radio_upfeature1 = self.uplayer1(radio_downfeature4, radio_downfeature3)
        radio_upfeature2 = self.uplayer2(radio_upfeature1, radio_downfeature2)
        radio_upfeature3 = self.uplayer3(radio_upfeature2, radio_downfeature1)
        radio_upfeature4 = self.uplayer4(radio_upfeature3, radio_feature)
        radio_upsample = self.upsample(radio_upfeature4)
        audio_output = self.outlayer(radio_upsample)

        return audio_output


class unet_TSB(nn.Module):
    def __init__(self, ngf=32, input_nc=1, output_nc=1, rescale=True):
        super(unet_TSB, self).__init__()
        self.rescale = rescale
        self.time_downsample_ratio = 2 ** 3 # This number equals 2^{#encoder_blcoks}

        #initialize layers
        self.inlayer = in_conv(input_nc, ngf)
        self.tsb_down1 = TSB(input_dim=128, in_channel=ngf, kernel_size=5, middle_kernel_size=25)
        self.downlayer1 = down_conv(ngf, ngf*2)
        self.tsb_down2 = TSB(input_dim=64, in_channel=ngf*2, kernel_size=5, middle_kernel_size=25)
        self.downlayer2 = down_conv(ngf*2, ngf*4)
        self.tsb_down3 = TSB(input_dim=32, in_channel=ngf*4, kernel_size=3, middle_kernel_size=15)
        self.downlayer3 = down_conv(ngf*4, ngf*8)

        self.uplayer2 = up_conv(ngf*8, ngf*4)
        self.uplayer3 = up_conv(ngf*4, ngf*2)
        self.uplayer4 = up_conv(ngf*2, ngf)
        self.outlayer = out_conv(ngf, output_nc)

        if self.rescale:
            rescale_module(self)

    def forward(self, radio_input):
        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = radio_input.size(-1)
        if not origin_len % self.time_downsample_ratio == 0:
            pad_len = int(np.ceil(origin_len / self.time_downsample_ratio)) \
                      * self.time_downsample_ratio - origin_len
            radio_input = F.pad(radio_input, pad=(0, pad_len, 0, 0))  #(bs, c, freq_bins, padded_time_steps, )

        radio_feature = self.inlayer(radio_input)
        radio_feature_tsb = self.tsb_down1(radio_feature)
        radio_downfeature1 = self.downlayer1(radio_feature_tsb)
        radio_downfeature1_tsb = self.tsb_down2(radio_downfeature1)
        radio_downfeature2 = self.downlayer2(radio_downfeature1_tsb)
        radio_downfeature2_tsb = self.tsb_down3(radio_downfeature2)
        radio_downfeature3 = self.downlayer3(radio_downfeature2_tsb)
        radio_upfeature2 = self.uplayer2(radio_downfeature3, radio_downfeature2)
        radio_upfeature3 = self.uplayer3(radio_upfeature2, radio_downfeature1)
        radio_upfeature4 = self.uplayer4(radio_upfeature3, radio_feature)
        audio_output = self.outlayer(radio_upfeature4)

        if not origin_len % self.time_downsample_ratio == 0:
            audio_output = audio_output[..., : origin_len] # (bs, channels, F, T)

        assert audio_output.size(-1) == origin_len

        return audio_output

class FTB(nn.Module):
    def __init__(self, input_dim=80, in_channel=32, r_channel=5):
        super(FTB, self).__init__()
        self.in_channel = in_channel
        self.freq_fc = nn.Linear(input_dim, input_dim, bias=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(in_channel),
            nn.ReLU())

    def forward(self, inputs):
        att_out = inputs
        att_out = torch.transpose(att_out, 2, 3)
        freqfc_out = self.freq_fc(att_out)
        att_out = torch.transpose(freqfc_out, 2, 3)

        cat_out = torch.cat([att_out, inputs], 1)
        outputs = self.conv2(cat_out)

        return outputs

class TSB(nn.Module):
    def __init__(self, input_dim=80, in_channel=32, kernel_size=5, middle_kernel_size=25):
        super(TSB, self).__init__()

        self.amp_conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=(kernel_size, kernel_size),
                      padding=(kernel_size//2, kernel_size//2)),
            nn.BatchNorm2d(in_channel),
            nn.ReLU())
        self.amp_conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=(1, middle_kernel_size), padding=(0, middle_kernel_size//2)),
            nn.BatchNorm2d(in_channel),
            nn.ReLU())
        self.amp_conv3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=(kernel_size, kernel_size),
                      padding=(kernel_size//2, kernel_size//2)),
            nn.BatchNorm2d(in_channel),
            nn.ReLU())
        self.ftb2 = FTB(input_dim=input_dim, in_channel=in_channel)

    def forward(self, amp):
        amp_out2 = self.amp_conv1(amp)
        amp_out3 = self.amp_conv2(amp_out2)
        amp_out4 = self.amp_conv3(amp_out3)
        amp_out5 = self.ftb2(amp_out4)

        return amp_out5