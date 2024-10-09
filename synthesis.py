state_path = '/path/to/best.pt'
config_path = '/path/to/config.yaml'
vocoder_config_path = '/path/to/vocoder/config/path/config.yml'
vocoder_ckpt_path = '/path/to/vocoder/checkpoint.pkl'
h5_pth = '/path/to/vocoder/statistics/h5/file/stats.h5'
root_path = '/path/to/pickle/noise/dataset/'
save_path = '/path/to/saved/files/'

import h5py
import os
import glob
import pickle
import torch
import numpy as np
import soundfile as sf
import yaml
from sklearn.preprocessing import StandardScaler
from loguru import logger
from uspeech_enhancement_model.model import unet_model
from parallel_wavegan.utils import load_model

def read_hdf5(hdf5_name, hdf5_path):
    with h5py.File(hdf5_name, "r") as hdf5_file:
        hdf5_data = hdf5_file[hdf5_path][()]
    return hdf5_data

with open(config_path, 'rb') as f:
    config = yaml.safe_load(f)

model = unet_model.unet_model(config)

state_dict = torch.load(state_path)['unet_en_model']
key_to_remove = 'speech_unet.transformer.embeddings.position_embeddings'
if key_to_remove in state_dict:
    del state_dict[key_to_remove]
model.load_state_dict(state_dict, strict=False)
model.eval().cpu()

with open(vocoder_config_path) as f:
    pwgan_config = yaml.safe_load(f)

vocoder = load_model(vocoder_ckpt_path, pwgan_config)
vocoder.remove_weight_norm()
vocoder = vocoder.cuda().eval()

scaler = StandardScaler()
scaler.mean_ = read_hdf5(h5_pth, "mean")
scaler.scale_ = read_hdf5(h5_pth, "scale")
scaler.n_features_in_ = scaler.mean_.shape[0]

def create_windows(input_sequence, window_size, stride, temporal_axis=1):
    windows = []
    total_length = input_sequence.shape[temporal_axis]
    for start in range(0, total_length - window_size + 1, stride):
        end = start + window_size
        slices = [slice(None)] * input_sequence.ndim
        slices[temporal_axis] = slice(start, end)
        window = input_sequence[tuple(slices)]
        windows.append(window)
    return windows

window_size = 122 
stride = 122 

speaker_test_path = root_path

pickle_files = glob.glob(os.path.join(speaker_test_path, "*.pickle"))

for pickle_file in pickle_files:
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    saved_name = pickle_file.replace(root_path, save_path).replace('.pickle', '.wav')

    ultrasound_syn = data['ultrasound_syn'] # or data['ultrasound_phy]
    clean_speech = data['clean_speech']
    noisy_speech = data['noisy_speech']

    print(f"noisy_speech shape: {noisy_speech.shape}")
    print(f"ultrasound_syn shape: {ultrasound_syn.shape}")

    sample_length_noisy = noisy_speech.shape[1]
    sample_length_ultrasound = ultrasound_syn.shape[1]

    min_length = min(sample_length_noisy, sample_length_ultrasound)
    if min_length < window_size:
        logger.warning(f"Sample {pickle_file} is shorter than window size {window_size}. Skipping this sample.")
        continue

    if sample_length_noisy > min_length:
        noisy_speech = noisy_speech[:, :min_length, :]
    if sample_length_ultrasound > min_length:
        ultrasound_syn = ultrasound_syn[:, :min_length, :]

    noisy_speech_windows = create_windows(noisy_speech, window_size, stride, temporal_axis=1)
    ultrasound_syn_windows = create_windows(ultrasound_syn, window_size, stride, temporal_axis=1)

    if len(noisy_speech_windows) != len(ultrasound_syn_windows):
        logger.warning(f"Mismatch in number of windows for sample {pickle_file}. Skipping.")
        continue

    if len(noisy_speech_windows) == 0:
        logger.warning(f"No valid windows for sample {pickle_file}. Skipping.")
        continue

    enhanced_speech_windows = []

    for noisy_window, ultrasound_window in zip(noisy_speech_windows, ultrasound_syn_windows):
        noisy_speech_input = torch.tensor(noisy_window).float()
        ultrasound_syn_input = torch.tensor(ultrasound_window).float()

        noisy_speech_input = noisy_speech_input.unsqueeze(0)
        ultrasound_syn_input = ultrasound_syn_input.unsqueeze(0)

        noisy_speech_input = noisy_speech_input.cpu()
        ultrasound_syn_input = ultrasound_syn_input.cpu()

        with torch.no_grad():
            output = model(noisy_speech_input, ultrasound_syn_input)['enhancement_speech']
            enhanced_speech = output.squeeze().cpu().numpy()
            enhanced_speech_windows.append(enhanced_speech)

    if not enhanced_speech_windows:
        logger.warning(f"No valid windows for sample {pickle_file}. Skipping.")
        continue

    enhanced_speech = np.concatenate(enhanced_speech_windows, axis=0)

    audio_input = scaler.transform(enhanced_speech)  
    audio_input = torch.tensor(audio_input).to('cuda')  

    with torch.no_grad():
        audio_pred = vocoder.inference(audio_input, normalize_before=False).view(-1).cpu().numpy()

    if not os.path.exists(os.path.dirname(saved_name)):
        os.makedirs(os.path.dirname(saved_name))

    sf.write(saved_name, audio_pred, config['audio_fs'])
