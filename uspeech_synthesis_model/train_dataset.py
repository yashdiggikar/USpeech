import glob
import os
import pickle
import torch
from torch.utils.data import Dataset

class collectedDataset(Dataset):
    def __init__(self, phases, config, p=1):
        self.config = config
        self.phases = phases

        self.user_list = config['user_list']
        self.data_list = []

        for user in self.user_list:
            for phase in phases:
                pickle_files = glob.glob(os.path.join(config['pickle_collected_dataset_dir'], user, phase, '*.pickle'))
                self.data_list += pickle_files

        self.data_list = self.data_list[:int(p*len(self.data_list))]

    def __getitem__(self, idx):
        data_path = self.data_list[idx]
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        filename = data['filename']
        audio_mel_segments = data['audio']
        ultrasound_doppler_segments = data['ultrasound']

        batch_audio = torch.FloatTensor(audio_mel_segments)
        batch_ultrasound = torch.FloatTensor(ultrasound_doppler_segments)

        result = {
            'filename': filename,
            'batch_audio': batch_audio,
            'batch_ultrasound': batch_ultrasound
        }
        return result

    def __len__(self):
        return len(self.data_list)