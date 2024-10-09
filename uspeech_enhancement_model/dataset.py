import glob
import os
import pickle
import torch
from loguru import logger
from torch.utils.data import Dataset


class unetDataset(Dataset):
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

        logger.info(f"Initializing dataset with {len(self.data_list)} items.")
    def __getitem__(self, idx):
        data_path = self.data_list[idx]
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        ultrasound_syn = data['ultrasound_syn']
        ultrasound_phy = data['ultrasound_phy']
        clean_speech = data['clean_speech']
        noisy_speech = data['noisy_speech']

        batch_ultrasound_syn = torch.FloatTensor(ultrasound_syn)
        batch_clean_speech = torch.FloatTensor(clean_speech)
        batch_noisy_speech = torch.FloatTensor(noisy_speech)
        batch_ultrasound_phy = torch.FloatTensor(ultrasound_phy)

        result = {
            'filename': data_path,
            'batch_ultrasound_syn': batch_ultrasound_syn,
            'batch_clean_speech': batch_clean_speech,
            'batch_noisy_speech': batch_noisy_speech,
            'batch_ultrasound_phy': batch_ultrasound_phy
        }

        return result

    def __len__(self):
        return len(self.data_list)