import sys

sys.path.append('../')
from pydub import AudioSegment
from settings import *
import pandas as pd
import numpy as np
from utils import create_folder
import librosa
from settings import *

ORIGINAL_SAMPLING_RATE = 44100
downsampled_output_path = os.path.join(MUSDB_WAVS_FOLDER_PATH + '_' + str(TARGET_SAMPLING_RATE))

folder_types = ['test', 'train']
for folder_type in folder_types:
    folder_type_path = os.path.join(MUSDB_WAVS_FOLDER_PATH, folder_type)
    folders = os.listdir(folder_type_path)
    for folder in folders:
        folder_path = os.path.join(folder_type_path, folder)
        files = os.listdir(folder_path)
        mixture, _ = librosa.load(os.path.join(folder_path, 'mixture.wav'), sr=ORIGINAL_SAMPLING_RATE)
        mix_stft = librosa.stft(mixture, win_length=NFFT, n_fft=NFFT, hop_length=HOP_LENGTH)

        mixture_down, _ = librosa.load(os.path.join(folder_path, 'mixture.wav'), sr=TARGET_SAMPLING_RATE)
        mix_stft_down = librosa.stft(mixture_down, win_length=NFFT, n_fft=NFFT, hop_length=HOP_LENGTH)

        for file in files:
            file_path = os.path.join(folder_path, file)
            y_ori, _ = librosa.load(file_path, sr=ORIGINAL_SAMPLING_RATE)
            y_down, fs = librosa.load(file_path, sr=TARGET_SAMPLING_RATE)

            # DOWNSAMPLED OUTPUT
            downsampled_folder_path = os.path.join(downsampled_output_path, folder_type, folder)
            create_folder(downsampled_folder_path)
            librosa.output.write_wav(os.path.join(downsampled_folder_path, file), y_down, sr=TARGET_SAMPLING_RATE)
