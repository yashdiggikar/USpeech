import os
import numpy as np
from pypesq import pesq
from pystoi import stoi
import librosa
import yaml
from loguru import logger
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

with open('/path/to/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

clean_dataset_path = '/path/to/the/clean/dataset/'
noisy_dataset_path = '/path/to/the/noisy/dataset/'
text_name = 'name_of_the_text_file.txt'

def alignment(ref_signal, deg_signal):
    distance, path = fastdtw(ref_signal, deg_signal, dist=euclidean)
    return distance, path

def calculate_lsd(ref_signal, deg_signal):
    def get_power(x):
        S = librosa.stft(x, win_length=2048, hop_length=512)
        S = np.log10(np.abs(S) ** 2 + 1e-8)
        return S

    s1 = get_power(ref_signal)
    s2 = get_power(deg_signal)
    lsd = np.mean(np.sqrt(np.mean((s1 - s2) ** 2, axis=1)))
    return lsd

def calculate_metrics(clean_file_path, noisy_file_path, sr=16000):
    clean_signal, _ = librosa.load(clean_file_path, sr=sr)
    noisy_signal, _ = librosa.load(noisy_file_path, sr=sr)

    if clean_signal.shape[0] != noisy_signal.shape[0]:
        start_idx = 5040
        end_idx = start_idx + len(noisy_signal)
        if end_idx <= len(clean_signal):
            clean_signal = clean_signal[start_idx:end_idx]
        else:
            clean_signal = clean_signal[start_idx:]
            min_length = min(len(clean_signal), len(noisy_signal))
            clean_signal = clean_signal[:min_length]
            noisy_signal = noisy_signal[:min_length]

    pesq_score = np.nan
    stoi_score = np.nan
    lsd_score = np.nan

    try:
        pesq_score = pesq(clean_signal, noisy_signal)
    except Exception as e:
        logger.info(f"Error calculating PESQ for {noisy_file_path}: {e}")

    try:
        stoi_score = stoi(clean_signal, noisy_signal, sr, extended=False)
    except Exception as e:
        logger.info(f"Error calculating STOI for {noisy_file_path}: {e}")
        
    try:
        lsd_score = calculate_lsd(clean_signal, noisy_signal)
    except Exception as e:
        logger.info(f"Error calculating LSD for {noisy_file_path}: {e}")

    duration = len(clean_signal) / sr

    return pesq_score, stoi_score, lsd_score, duration

with open(text_name, 'w') as text_file:
    results = {}
    for speaker in ['']:
        speaker_metrics = {'pesq': [], 'stoi': [], 'lsd': []}
        for set_type in ['']:
            clean_set_path = os.path.join(clean_dataset_path, speaker, set_type)
            noisy_set_path = os.path.join(noisy_dataset_path, speaker, set_type)
            if '-5' in noisy_set_path or '0' in noisy_set_path:
                continue
            for filename in os.listdir(clean_set_path):
                clean_file_path = os.path.join(clean_set_path, filename)
                base_filename, _ = os.path.splitext(filename)
                noisy_filenames = [f for f in os.listdir(noisy_set_path) if f.startswith(base_filename + '_')]
                for noisy_filename in noisy_filenames:
                    noisy_file_path = os.path.join(noisy_set_path, noisy_filename)
                    try:
                        scores = calculate_metrics(clean_file_path, noisy_file_path, 16000)
                        metrics = scores[:-1]
                        duration = scores[-1]

                        if any(np.isnan(metric) for metric in metrics):
                            logger.info(f"One of the metrics is NaN for {noisy_file_path}. Skipping this sample.")
                            continue

                        pesq_score, stoi_score, lsd_score = metrics
                        speaker_metrics['pesq'].append(pesq_score)
                        speaker_metrics['stoi'].append(stoi_score)
                        speaker_metrics['lsd'].append(lsd_score)

                        text_file.write(f"{clean_file_path}, {noisy_file_path}, duration: {duration:.2f}, pesq: {pesq_score}, stoi: {stoi_score}, lsd: {lsd_score}\n")

                    except Exception as e:
                        logger.info(f"Error processing file {noisy_filename}: {e}")

        for metric in speaker_metrics:
            speaker_metrics[metric] = np.mean(speaker_metrics[metric]) if speaker_metrics[metric] else None
        results[speaker] = speaker_metrics

    overall_metrics = {'pesq': [], 'stoi': [], 'lsd': []}
    for speaker, metrics in results.items():
        for metric in overall_metrics:
            if metrics[metric] is not None:
                overall_metrics[metric].append(metrics[metric])

    for metric in overall_metrics:
        overall_metrics[metric] = np.mean(overall_metrics[metric]) if overall_metrics[metric] else None

    text_file.write("\nIndividual Speaker Metrics:\n")
    for speaker, metrics in results.items():
        text_file.write(f"{speaker}:\n")
        for metric, value in metrics.items():
            text_file.write(f"  {metric}: {value}\n")
        text_file.write("\n")
    text_file.write("Overall Metrics:\n")
    for metric, value in overall_metrics.items():
        text_file.write(f"{metric}: {value}\n")
