import yaml
import numpy as np
import os
import random
from torch.utils.data import Dataset, DataLoader
import librosa
import pickle
from tqdm import tqdm
from loguru import logger
import soundfile as sf
import torch
from uspeech_synthesis_model.modules.ultrasoundAudioModel import ultrasoundAudioModel
from pathlib import Path


class PickleNoiseCollectedDataset(Dataset):
    def __init__(self, config):
        self.list = []
        self.root_dir = config['clean_collected_dataset_dir']
        self.new_root_dir = config['noisy_collected_dataset_dir']
        self.pickle_cllected_dataset_dir = config['pickle_collected_dataset_dir']
        
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.wav'):
                    full_path = os.path.join(subdir, file)
                    if 'test' in full_path:
                      self.list.append(full_path)
        
        logger.info(f"Initializing dataset with {len(self.list)} items.")
    
    def __getitem__(self, idx):

        filename = self.list[idx]
        savefilename = filename.replace(self.root_dir, self.new_root_dir).replace('.wav', '.pickle')
        pickle_saved_dir = os.path.dirname(savefilename)

        save_path = Path(savefilename)
        relative_path = save_path.relative_to(self.new_root_dir)

        ultrasound_phy_path = Path(self.pickle_collected_dataset_dir) / relative_path
        with open(ultrasound_phy_path, 'rb') as pf:
            ultrasound_phy = pickle.load(pf)['ultrasound']

        log_clean_speech_mel_spec, ultrasound_syn, noisy_speech_list, output_filename_list = self.get_the_paired_data(filename)

        result_dict_list = []

        for i, noisy_speech in enumerate(noisy_speech_list):
            noisy_speech = np.nan_to_num(noisy_speech)
            log_noisy_speech_mel_spec = self.log_mel_spectrogram(noisy_speech, config['audio_fs'])
            log_noisy_speech_mel_spec = np.delete(log_noisy_speech_mel_spec, np.r_[0:32, -32:0], axis=1)
            log_noisy_speech_mel_spec = np.expand_dims(np.transpose(log_noisy_speech_mel_spec), axis=0)

            result_dict = {
                'clean_speech': log_clean_speech_mel_spec,
                'noisy_speech': log_noisy_speech_mel_spec,
                'ultrasound_syn': ultrasound_syn,
                'ultrasound_phy': ultrasound_phy
            }
            result_dict_list.append(result_dict)
            os.makedirs(pickle_saved_dir, exist_ok=True)
            output_pickle_path = os.path.join(
                pickle_saved_dir,
                output_filename_list[i].replace('.wav', '.pickle')
            )
            with open(output_pickle_path, 'wb') as f:
                pickle.dump(result_dict, f)

        return result_dict_list


    def __len__(self):
        return len(self.list)
    
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

    def mix_noise(self, speech, noise, snr_db):
        speech_power = np.sum(speech ** 2)
        noise_power = np.sum(noise ** 2)

        snr_linear = 10 ** (snr_db / 10)
        scaling_factor = np.sqrt(speech_power / (snr_linear * noise_power))

        noisy_speech = speech + scaling_factor * noise
        noisy_speech = np.clip(noisy_speech, -1, 1)

        return noisy_speech
    
    def synthesize_noisy_speech(self, clean_speech_path, noise_files, output_dir, num_samples=1):
        clean_speech, sr = librosa.load(clean_speech_path, sr=config['audio_fs'])
        clean_speech_name = os.path.splitext(os.path.basename(clean_speech_path))[0]

        os.makedirs(output_dir, exist_ok=True)

        noisy_speech_list = []
        output_filename_list = []

        for _ in range(num_samples):
            noise_file = random.choice(noise_files)
            noise_name = os.path.splitext(os.path.basename(noise_file))[0]
            noise_path = os.path.join(config['noise_source_dir'], noise_file)
            noise, _ = librosa.load(noise_path, sr=sr)

            if len(noise) < len(clean_speech):
                noise = np.tile(noise, int(np.ceil(len(clean_speech) / len(noise))))[:len(clean_speech)]
            else:
                noise = noise[:len(clean_speech)]

            snr_db = random.choice([-5, 0, 5, 10, 15])
            noisy_speech = self.mix_noise(clean_speech, noise, snr_db)
            noisy_speech_list.append(noisy_speech)

            output_filename = f"{clean_speech_name}_{noise_name}_snr_{snr_db}.wav"
            output_filename_list.append(output_filename)
            output_path = os.path.join(os.path.dirname(clean_speech_path.replace('clean_', 'noisy_')), output_filename)
            
            os.makedirs(os.path.dirname(clean_speech_path.replace('clean_', 'noisy_')), exist_ok=True)

            sf.write(output_path, noisy_speech, sr)
        return noisy_speech_list, output_filename_list
    
    def get_the_paired_data(self, filename):
        clean_speech, _ = librosa.load(filename, sr=config['audio_fs'])
        log_clean_speech_mel_spec = self.log_mel_spectrogram(clean_speech, config['audio_fs'])
        log_clean_speech_mel_spec = np.delete(log_clean_speech_mel_spec, np.r_[0:32, -32:0], axis=1)
        log_clean_speech_mel_spec = np.expand_dims(np.transpose(log_clean_speech_mel_spec), axis=0)
        input_audio = torch.tensor(log_clean_speech_mel_spec).cpu().float().unsqueeze(0)
        ultrasound_syn = model(input_audio)['ultrasound_syn'].detach().squeeze(0).numpy()
        noisy_speech_list, output_filename_list = self.synthesize_noisy_speech(filename, noise_files, config['noisy_processed_collected_dataset_dir'])

        return log_clean_speech_mel_spec, ultrasound_syn, noisy_speech_list, output_filename_list
    
        
if __name__ == '__main__':
    with open('/path/to/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    noise_files = [file for file in os.listdir(config['noise_source_dir']) if file.endswith('.wav')]
    model = ultrasoundAudioModel(config).cpu()
    dict_path = config['synthesis_model']
    state_dict = torch.load(dict_path)
    model.load_state_dict(state_dict['model'])
    model.eval()

    dataset = PickleNoiseCollectedDataset(config)
    loader = DataLoader(dataset,
            batch_size = 1,
            num_workers = 0,
            shuffle = False,
            drop_last = False)
    
    for i, data in enumerate(tqdm(loader, desc="Processing", unit="batch")):
        pass