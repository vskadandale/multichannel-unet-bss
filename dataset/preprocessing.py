import numpy as np
import sys

sys.path.append('../')
from functools import partial
import librosa
import torch
from utils import create_folder
import librosa.display
from sklearn.model_selection import train_test_split
import shutil
import csv
from settings import *


def get_sources(folder):
    y = [librosa.load(os.path.join(folder, element + '.wav'),
                      sr=ORIGINAL_SAMPLING_RATE)[0] for element in [*SOURCES, 'mixture']]
    downsampled = np.stack(list(map(partial(librosa.core.resample,
                                            orig_sr=ORIGINAL_SAMPLING_RATE,
                                            target_sr=TARGET_SAMPLING_RATE), y)))
    return downsampled


def split_sources(sources, flag):
    window = int(TARGET_SAMPLING_RATE * DURATION)
    M = (np.max(sources.shape) // window)
    if flag == 'train':
        splits = sources[..., :M * window]
        splits = np.reshape(splits, splits.shape[:-1] + (M, window))
    else:
        zero_padding = window - np.max(sources.shape) % window
        splits = np.concatenate([sources, np.zeros([len(SOURCES) + 1, zero_padding])], axis=1)
        splits = np.reshape(splits, splits.shape[:-1] + (M + 1, window))
    return splits


def _stft(sources):
    s = torch.from_numpy(sources).float().cuda(cuda)
    shape = s.size()
    with torch.no_grad():
        stft_output = torch.stft(s.view(-1, shape[-1]),
                                 n_fft=NFFT,
                                 hop_length=HOP_LENGTH,
                                 window=torch.hann_window(NFFT).cuda(cuda))
        stft = stft_output.view(*shape[:-1], *stft_output.size()[1:3], 2).data.cpu().numpy()
    stft = stft[..., 0] + stft[..., 1] * 1j
    return stft


def get_signal_energy(signal):
    return sum(abs(signal) ** 2)


def save_chunks(chunk_id, subset_type, track_name, sources, energy_profile):
    save_folder_path = os.path.join(CHUNKS_PATH, subset_type, track_name, str(chunk_id))
    create_folder(save_folder_path)
    true_label = np.zeros(len(SOURCES) + 1, dtype='int')
    for source_id, source in enumerate([*SOURCES, 'MIX']):
        signal = sources[source_id, chunk_id]
        signal_energy = get_signal_energy(signal)
        if int(signal_energy) > ENERGY_THRESHOLD:
            true_label[source_id] = 1
        save_path = os.path.join(save_folder_path, source + '_' + str(int(round(signal_energy))) + '.wav')
        energy_profile[source_id][os.path.dirname(save_path)] = signal_energy
        librosa.output.write_wav(save_path, signal, TARGET_SAMPLING_RATE)
    return energy_profile, true_label[:-1]


VALIDATION_PATH = os.path.join(MUSDB_SPLITS_PATH, 'val')
train_paths = []
energy_profile = [{} for _ in range(len(SOURCES) + 1)]
sample_dict = {}
subset = ['train', 'test']
cuda = 0

for subset_type in subset:
    DATA_PATH = os.path.join(MUSDB_WAVS_FOLDER_PATH, subset_type)
    tracks = sorted(os.listdir(DATA_PATH))

    for track_id, track_name in enumerate(tracks):
        track_path = os.path.join(DATA_PATH, track_name)
        dump_path = os.path.join(MUSDB_SPLITS_PATH, subset_type, track_name)
        create_folder(dump_path)
        sources_downsampled = get_sources(track_path)
        sources_split = split_sources(sources_downsampled, subset_type)
        stft_output = _stft(sources_split)
        for chunk_id in range(stft_output.shape[1]):
            matrix = stft_output[:, chunk_id, ...]
            energy_profile, true_label = save_chunks(chunk_id, subset_type, track_name, sources_split, energy_profile)
            if subset_type == 'train':
                train_paths.append(os.path.join(dump_path, str(chunk_id) + '.npy'))
            sample_dict['spec'] = matrix
            sample_dict['true_label'] = true_label
            full_path = os.path.join(dump_path, str(chunk_id))
            np.save(full_path, sample_dict)
            print('[{0}/{1}] || [{2}||{3}]'.format(chunk_id + 1, stft_output.shape[1], track_id + 1, len(tracks)))
        del sources_split

create_folder(ENERGY_PROFILE_FOLDER)
for source_id, source in enumerate([*SOURCES, 'MIX']):
    with open(os.path.join(ENERGY_PROFILE_FOLDER, source + '_energy_profile.csv'), 'w') as f:
        w = csv.writer(f)
        w.writerows(energy_profile[source_id].items())
    np.save(os.path.join(ENERGY_PROFILE_FOLDER, source + '_energy_profile'), energy_profile[source_id])

########## CREATING TRAINING-VALIDATION SPLIT##############
X_train, X_val = train_test_split(train_paths, test_size=0.05, random_state=0)
create_folder(VALIDATION_PATH)

for file in X_val:
    val_path = str.replace(file, 'train', 'val')
    create_folder(os.path.abspath(os.path.join(val_path, os.pardir)))
    shutil.move(file, val_path)
