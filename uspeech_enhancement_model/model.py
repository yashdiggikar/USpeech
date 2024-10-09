import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.bottleneck_transformer import *
from .modules.UNet import *

class en_down_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(en_down_conv, self).__init__()
        self.downconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1,3), stride=2, padding=(4,1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1,3), padding=(2,1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True))

    def forward(self, downconv_input):
        return self.downconv(downconv_input)
    
class ultrasoundEncoder(nn.Module):
    def __init__(self, ngf=32, input_nc=1, rescale=True):
        super(ultrasoundEncoder, self).__init__()
        self.rescale = rescale
        self.time_downsample_ratio = 2 ** 3 # This number equals 2^{#encoder_blcoks}

        self.inlayer = in_conv(input_nc, ngf)
        self.downlayer1 = en_down_conv(ngf, ngf*2)
        self.downlayer2 = en_down_conv(ngf*2, ngf*4)
        self.downlayer3 = en_down_conv(ngf*4, ngf*8)

        if self.rescale:
            rescale_module(self)
        
    def forward(self, ultrasound_input):
        ultrasound_input = ultrasound_input.permute(0,1,3,2).contiguous()
        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = ultrasound_input.size(-1)
        if not origin_len % self.time_downsample_ratio == 0:
            pad_len = int(np.ceil(origin_len / self.time_downsample_ratio)) \
                      * self.time_downsample_ratio - origin_len
            ultrasound_input = F.pad(ultrasound_input, pad=(0, pad_len, 0, 0))  #(bs, c, freq_bins, padded_time_steps)

        inlayer_output = self.inlayer(ultrasound_input)
        downlayer1_output = self.downlayer1(inlayer_output)
        downlayer2_output = self.downlayer2(downlayer1_output)
        downlayer3_output = self.downlayer3(downlayer2_output)

        return downlayer3_output


class speechUnet(nn.Module):
    def __init__(self, config, ngf=32, input_nc=1, output_nc=1, rescale=True):
        super(speechUnet, self).__init__()
        self.config = config
        self.rescale = rescale
        self.time_downsample_ratio = 2 ** 3 # This number equals 2^{#encoder_blcoks}

        self.inlayer = in_conv(input_nc, ngf)
        self.tsb_down1 = TSB(input_dim=128, in_channel=ngf, kernel_size=5, middle_kernel_size=25)
        self.downlayer1 = down_conv(ngf, ngf*2)
        self.tsb_down2 = TSB(input_dim=64, in_channel=ngf*2, kernel_size=5, middle_kernel_size=25)
        self.downlayer2 = down_conv(ngf*2, ngf*4)
        self.tsb_down3 = TSB(input_dim=32, in_channel=ngf*4, kernel_size=3, middle_kernel_size=15)
        self.downlayer3 = down_conv(ngf*4, ngf*8)

        self.transformer = Transformer(self.config)
        self.postlayer = PostLayer(self.config)

        self.converter = nn.Sequential(
            nn.Conv2d(
                in_channels=ngf*16,
                out_channels=ngf*8,
                kernel_size=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ngf*8)
        )

        self.uplayer2 = up_conv(ngf*8, ngf*4)
        self.uplayer3 = up_conv(ngf*4, ngf*2)
        self.uplayer4 = up_conv(ngf*2, ngf)
        self.outlayer = out_conv(ngf, output_nc)

        if self.rescale:
            rescale_module(self)

    def forward(self, speech_input, ultrasound_condition):
        speech_input = speech_input.permute(0,1,3,2).contiguous()
        origin_len = speech_input.size(-1)
        if not origin_len % self.time_downsample_ratio == 0:
            pad_len = int(np.ceil(origin_len / self.time_downsample_ratio)) \
                      * self.time_downsample_ratio - origin_len
            speech_input = F.pad(speech_input, pad=(0, pad_len, 0, 0))  #(bs, c, freq_bins, padded_time_steps, )
        
        inlayer_output = self.inlayer(speech_input)
        speech_feature_tsb = self.tsb_down1(inlayer_output)
        speech_downfeature1 = self.downlayer1(speech_feature_tsb)
        speech_downfeature1_tsb = self.tsb_down2(speech_downfeature1)
        speech_downfeature2 = self.downlayer2(speech_downfeature1_tsb)
        speech_downfeature2_tsb = self.tsb_down3(speech_downfeature2)
        speech_downfeature3 = self.downlayer3(speech_downfeature2_tsb)
        speech_downfeature3 = torch.cat([speech_downfeature3, ultrasound_condition], dim=1)

        speech_downfeature3 = self.transformer(speech_downfeature3)
        speech_downfeature3 = self.postlayer(speech_downfeature3)
        speech_downfeature3 = self.converter(speech_downfeature3)

        speech_upfeature2 = self.uplayer2(speech_downfeature3, speech_downfeature2)
        speech_upfeature3 = self.uplayer3(speech_upfeature2, speech_downfeature1)
        speech_upfeature4 = self.uplayer4(speech_upfeature3, inlayer_output)
        enhancement_output = self.outlayer(speech_upfeature4)

        if not origin_len % self.time_downsample_ratio == 0:
            enhancement_output = enhancement_output[..., : origin_len] # (bs, channels, F, T)

        assert enhancement_output.size(-1) == origin_len
        return enhancement_output.permute(0,1,3,2).contiguous()

class unet_model(nn.Module):
    def __init__(self, config, ngf=32, input_nc=1, output_nc=1, rescale=True):
        super(unet_model, self).__init__()
        self.config = config
        self.ultrasound_encoder = ultrasoundEncoder(ngf=ngf, input_nc=input_nc, output_nc=output_nc, rescale=rescale)
        self.speech_unet = speechUnet(config, ngf=ngf, input_nc=input_nc, output_nc=output_nc, rescale=rescale)

    def forward(self, speech_input, ultrasound_input):
        ultrasound_condition = self.ultrasound_encoder(ultrasound_input)
        enhancement_output = self.speech_unet(speech_input, ultrasound_condition)
        result = {
            'enhancement_speech': enhancement_output,
        }
        return result
        

def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        return {
            'Total': total_params,
            'Trainable': trainable_params,
            'Non-trainable': non_trainable_params
        }