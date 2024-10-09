import torch.nn as nn
import yaml

from modules.pann import Cnn14_unet
from ultrasoundDecoder import UNetDecoder

with open('/path/to/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class ultrasoundAudioModel(nn.Module):

    def __init__(self, config):
        super(ultrasoundAudioModel, self).__init__()
        self.config = config
        self.audio_encoder = Cnn14_unet(embed_dim=512, pretrained=True)
        self.ultrasound_decoder = UNetDecoder()

    def forward(self, audio):
        _, encoder_features = self.audio_encoder(audio)
        ultrasound_syn = self.ultrasound_decoder(encoder_features[-1], encoder_features)
        result = {
            'ultrasound_syn': ultrasound_syn
        }
        return result
    