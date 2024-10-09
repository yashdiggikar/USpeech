import glob
import os
import pickle
import random
import torch
from torch.utils.data import Dataset

class videoAudioDataset(Dataset):
    def __init__(self, phases, config):
        self.config = config
        self.phases = phases
        self.labels = sorted(os.listdir(config['pickle_lrw_dir']))

        self.data_list = []

        for label in self.labels:
            for phase in phases:
                pickle_files = glob.glob(os.path.join(config['pickle_lrw_dir'], label, phase, '*.pickle'))
                self.data_list += pickle_files
        random.shuffle(self.data_list)
    
    def __getitem__(self, idx):
        data_path = self.data_list[idx]
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        filename = data['filename']
        label = data['label']
        audio_mel_segments = data['audio']
        video_color_segments = data['video']

        batch_audio = audio_mel_segments
        batch_video = video_color_segments

        batch_audio = torch.FloatTensor(batch_audio)
        batch_video = torch.FloatTensor(batch_video)
        batch_video = batch_video.permute(1, 0, 2, 3)
        
        result = {
            'filename': filename,
            'label': label,
            'batch_audio': batch_audio,
            'batch_video': batch_video
        }
        return result
    
    def __len__(self):
        return len(self.data_list)