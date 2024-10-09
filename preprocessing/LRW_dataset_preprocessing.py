import yaml
import numpy as np
import glob
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import librosa
import pickle
from tqdm import tqdm
from loguru import logger

class PickleLRWDataset(Dataset):
    def __init__(self, config):
        self.config = config

        with open(self.config['label_sorted']) as myfile:
            self.labels = myfile.read().splitlines()
        self.list = []
        self.target_dir = config['pickle_lrw_dir_name']

        for (i, label) in enumerate(self.labels):
            files = glob.glob(os.path.join(self.config['lrw_dir'], label, '*', '*.mp4'))
            for file in files:
                savefile = file.replace('lipread_mp4', self.target_dir).replace('.mp4', '.pickle')
                savepath = os.path.split(savefile)[0]
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
            files = sorted(files)
            self.list += [(file, i) for file in files]
        logger.info(f"Initializing dataset with {len(self.list)} items.")
    
    def __getitem__(self, idx):
        filename = self.list[idx][0]
        label = self.list[idx][1]
        normalized_frames, normalized_log_mel_spec = self.get_the_paired_data(filename)
        result_dict = {
            'filename': filename,
            'label': label,
            'video': normalized_frames,
            'audio': normalized_log_mel_spec
        }
        savefilename = filename.replace('lipread_mp4', self.target_dir).replace('.mp4', '.pickle')
        with open(savefilename, 'wb') as f:
            pickle.dump(result_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        return result_dict

    def __len__(self):
        return len(self.list)

    def extract_frames(self, filename):
        cap = cv2.VideoCapture(filename)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = frame[99:227, 63:191]  # Cropping the frame 128 * 128
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
                frames.append(frame)
            else:
                break
        cap.release()
        frames = np.stack(frames)
        frames = np.append(frames, frames[-1][np.newaxis, ...], axis=0)
        return frames

    def log_mel_spectrogram(self, raw_audio, sr, spec_power=1):
        mel_baisis = librosa.filters.mel(sr=sr, n_fft=config['mel_nfft'], n_mels=config['n_mels'],
                                        fmin=config['mel_fmin'], fmax=config['mel_fmax'])
        spec = np.abs(librosa.stft(raw_audio, n_fft=config['mel_nfft'], hop_length=config['mel_hop_length'])) ** spec_power
        mel_spec = np.dot(mel_baisis, spec)
        mel_spec = np.maximum(1e-5, mel_spec)
        mean = np.mean(mel_spec)
        variance = np.var(mel_spec)
        std = np.sqrt(variance)
        log_mel_spec = (mel_spec - mean) / std
        log_mel_spec = np.log10(mel_spec)
        return log_mel_spec

    def get_the_paired_data(self, filename):
        frames = self.extract_frames(filename)
        frames = np.array(frames)
        audio, sample_rate = librosa.load(filename, sr=config['audio_fs'])
        log_mel_spec = self.log_mel_spectrogram(audio, sample_rate)
        normalized_frames, normalized_log_mel_spec = frames, log_mel_spec
        normalized_frames = np.transpose(normalized_frames, (0, 3, 1, 2))
        normalized_log_mel_spec = np.expand_dims(np.transpose(normalized_log_mel_spec), axis=0)
        return normalized_frames, normalized_log_mel_spec
    
if __name__ == '__main__':

    with open('/path/to/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    dataset = PickleLRWDataset(config)
    loader = DataLoader(dataset,
            batch_size = 128,
            num_workers = 32,
            shuffle = False,
            drop_last = False)
    
    for i, data in enumerate(tqdm(loader, desc="Processing", unit="batch")):
        pass