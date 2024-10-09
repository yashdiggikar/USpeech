from .modules.pann import Cnn14
from .modules.slowonly import Slowonly
import torch
import torch.nn as nn
import yaml
from loguru import logger

with open('/path/to/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class videoAudioModel(nn.Module):
    
    def __init__(self, config):
        super(videoAudioModel, self).__init__()
        self.config = config
        self.video_encoder = Slowonly(embed_dim=512, depth=50)
        self.audio_encoder = Cnn14(embed_dim=512, pretrained=True)
        self.video_adaptive_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.audio_adaptive_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, video, audio):
        video_temp_emb = self.video_encoder(video)
        audio_temp_emb = self.audio_encoder(audio)
        video_emb = self.video_adaptive_avg_pool(video_temp_emb.permute(0, 2, 1)).squeeze(-1)
        audio_emb = self.audio_adaptive_avg_pool(audio_temp_emb.permute(0, 2, 1)).squeeze(-1)

        result = {
            'video_temp_emb': video_temp_emb,
            'video_emb': video_emb,
            'audio_temp_emb': audio_temp_emb,
            'audio_emb': audio_emb
        }

        return result


def load_dict(model, config, check=False):
    video_encoder_dict = config['slowonly_pretrained']
    audio_encoder_dict = config['pann_pretrained']

    video_state_dict = torch.load(video_encoder_dict)['state_dict']
    new_video_state_dict = {}
    for key in video_state_dict.keys():
        if key.startswith("backbone."):
            new_key = key.replace("backbone.", "resnet3dslowonly.")
            new_video_state_dict[new_key] = video_state_dict[key]
    model.video_encoder.load_state_dict(new_video_state_dict, strict=False)

    audio_state_dict = torch.load(audio_encoder_dict)['model']
    model.audio_encoder.load_state_dict(audio_state_dict, strict=False)
    
    if check:
        result_video = model.video_encoder.load_state_dict(new_video_state_dict, strict=False)
        result_audio = model.audio_encoder.load_state_dict(audio_state_dict, strict=False)

        logger.info(f'All keys in video encoder: {list(new_video_state_dict.keys())}')
        logger.info(f'Missing keys in video encoder: {result_video.missing_keys}')
        logger.info(f'Unexpected keys in video encoder: {result_video.unexpected_keys}')

        logger.info(f'All keys in audio encoder: {list(audio_state_dict.keys())}')
        logger.info(f'Missing keys in audio encoder: {result_audio.missing_keys}')
        logger.info(f'Unexpected keys in audio encoder: {result_audio.unexpected_keys}')

    return model