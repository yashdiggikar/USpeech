import yaml
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import librosa
import pickle
from tqdm import tqdm
from loguru import logger
import scipy.signal as signal
import soundfile as sf

class PickleCollectedDataset(Dataset):
    def __init__(self, config):
        self.list = []
        self.root_dir = config['collected_dataset_dir']
        self.new_root_dir = config['pickle_collected_dataset_dir']
        self.clean_audio_dir = config['clean_audio_dir']

        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.WAV') or file.endswith('.wav'):
                    full_path = os.path.join(subdir, file)
                    self.list.append(full_path)
        
        logger.info(f"Initializing dataset with {len(self.list)} items.")
    
    def __getitem__(self, idx):
        filename = self.list[idx]
        savefilename = filename.replace(self.root_dir, self.new_root_dir).replace('.wav', '.pickle').replace('.WAV', '.pickle')
        log_audio_mel_spec, log_ultrasound_doppler_spec = self.get_the_paired_data(filename)
        result_dict = {
            'filename': filename,
            'ultrasound': log_ultrasound_doppler_spec,
            'audio': log_audio_mel_spec
        }
        try:
            os.makedirs(os.path.dirname(savefilename))
        except FileExistsError:
            pass

        with open(savefilename, 'wb') as f:
            pickle.dump(result_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return result_dict

    def __len__(self):
        return len(self.list)

    def low_filter(self, source, config):
        sos = signal.ellip(8, 1, 100, config['low_filter']['freq_low'], 'lowpass', 
                        fs=config['low_filter']['sampling_rate'], output='sos',
                            analog=config['low_filter']['analog'])
        filtered = signal.sosfilt(sos, source)
        return filtered

    def high_filter(self, source, config):
        sos = signal.ellip(8, 1, 100, config['high_filter']['freq_high'], 'highpass', 
                        fs=config['high_filter']['sampling_rate'], output='sos',
                            analog=config['high_filter']['analog'])
        filtered = signal.sosfilt(sos, source)
        return filtered

        
    def resampler(self, source, config):
        return librosa.resample(source, orig_sr=config['collected_fs'], target_sr=config['audio_fs'])

        
    def stft_ultra(self, source, config):
        f, t, Zxx = signal.stft(source, fs=config['ultrasound_fs'], 
                                nfft=config['ultrasound_nfft'], nperseg=config['ultrasound_win_length'], 
                                noverlap=config['ultrasound_overlap_length'])
        return f,t,Zxx


    def doppler_extractor(self, raw_ultrasound, config):
        '''doppler_extractor.
        Extract doppler shift feature from ultrasound.

        :param raw_ultrasound: raw ultrasound waveform
        :param normalization: doppler shift spectrum whether need normalization 

        additional info:
        original_freq = [17250, 18000, 18750, 19500, 20250, 21000, 21750, 22500]
        doppler_shift_list = np.array([
                -93.6, -81.9, -70.2, -58.5, -46.8, -35.1, -23.4, 
                23.4,  35.1,  46.8,  58.5,  70.2,  81.9,  93.6])
        '''
        
        _, _, uzxx = self.stft_ultra(raw_ultrasound, config=config)
        idx = np.array([1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920])
        doppler = [np.delete(uzxx[idx[x]-8:idx[x]+9, :], [7,8,9], axis=0) for x in range(len(idx))]
        doppler = np.stack(doppler, axis=1)
        doppler, _ = librosa.magphase(doppler)
        doppler = np.transpose(doppler, (1,2,0))
        doppler = doppler.mean(0)
        doppler = doppler.T
        doppler = librosa.amplitude_to_db(doppler, ref=np.max)
        return doppler


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

        
    def log_doppler_spectrogram(self, raw_ultrasound, config):
        return self.doppler_extractor(raw_ultrasound, config)


    def get_the_paired_data(self, filename):
        raw_signal, _ = librosa.load(filename, sr=config['collected_fs'])
        audio = self.resampler(self.low_filter(raw_signal, config), config)
        
        # save the clean audio, if needed
        sf.write(os.path.join(self.clean_audio_dir, os.path.basename(filename)), audio, config['audio_fs'])
        
        ultrasound = self.high_filter(raw_signal, config)
        log_audio_mel_spec = self.log_mel_spectrogram(audio, config['audio_fs'])
        log_ultrasound_doppler_spec = self.log_doppler_spectrogram(ultrasound, config['ultrasound_fs'], config)
        log_audio_mel_spec = np.delete(log_audio_mel_spec, np.r_[0:32, -32:0], axis=1)  # remove the beginning and ending 32 bins to avoid the spectrum leakage
        log_ultrasound_doppler_spec = np.delete(log_ultrasound_doppler_spec, np.r_[0:32, -32:0], axis=1)

        log_audio_mel_spec = np.expand_dims(np.transpose(log_audio_mel_spec), axis=0)
        log_ultrasound_doppler_spec = np.expand_dims(np.transpose(log_ultrasound_doppler_spec), axis=0)
        return log_audio_mel_spec, log_ultrasound_doppler_spec
        
if __name__ == '__main__':

    with open('/path/to/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    dataset = PickleCollectedDataset(config)
    loader = DataLoader(dataset,
            batch_size = 1,
            num_workers = 0,
            shuffle = False,
            drop_last = False)
    
    for i, data in enumerate(tqdm(loader, desc="Processing", unit="batch")):
        pass