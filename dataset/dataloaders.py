from pathlib import Path
import torch
import torch.utils.data
import numpy as np
import random
from utils.utils import get_conditions
from settings import *


class UnetInput(torch.utils.data.Dataset):
    def __init__(self, state):
        self.SPECTROGRAM_DIRECTORY = os.path.join(MUSDB_SPLITS_PATH, state)
        self.L = len(SOURCES_SUBSET)
        self.remove_source_ids = np.setdiff1d(np.arange(len(SOURCES)), SOURCES_SUBSET_ID)

        self.shortlisted = np.load(FILTERED_SAMPLE_PATHS + '.npy')
        self.input_list = []
        paths = list(Path(self.SPECTROGRAM_DIRECTORY).rglob("*.npy"))  ## Finds all the .npy files in the directory
        for filepath in paths:
            filepath_str = filepath.as_posix()
            if (filepath_str in self.shortlisted) or state != 'train':
                self.input_list.append(filepath_str)

        _ = random.shuffle(self.input_list)

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        sample = np.load(self.input_list[idx], allow_pickle=True).item()
        mixture_phase = np.angle(sample['spec'][-1])
        mags = np.absolute(np.nan_to_num(np.delete(sample['spec'], self.remove_source_ids, axis=0))) + np.finfo(
            np.float).eps
        true_label = np.delete(sample['true_label'], self.remove_source_ids, axis=0)
        return torch.from_numpy(mags).float(), \
               [torch.from_numpy(mixture_phase).unsqueeze(0), self.input_list[idx], torch.from_numpy(true_label)]


class CUnetInput(torch.utils.data.Dataset):
    def __init__(self, state):
        self.SPECTROGRAM_DIRECTORY = os.path.join(MUSDB_SPLITS_PATH, state)
        self.L = len(SOURCES_SUBSET)
        self.remove_source_ids = np.setdiff1d(np.arange(len(SOURCES)), SOURCES_SUBSET_ID)

        conditions = get_conditions(self.L, state)
        self.shortlisted = np.load(FILTERED_SAMPLE_PATHS + '.npy')
        self.input_list = []
        paths = list(Path(self.SPECTROGRAM_DIRECTORY).rglob("*.npy"))  ## Finds all the .npy files in the directory
        for filepath in paths:
            filepath_str = filepath.as_posix()
            if (filepath_str in self.shortlisted) or state != 'train':
                for condition in conditions:
                    self.input_list.append([filepath_str, condition])

        _ = random.shuffle(self.input_list)

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        file_name, condition = self.input_list[idx]
        sample = np.load(file_name, allow_pickle=True).item()
        mixture_phase = np.angle(sample['spec'][-1])
        mags = np.absolute(np.nan_to_num(np.delete(sample['spec'], self.remove_source_ids, axis=0))) + np.finfo(
            np.float).eps
        true_label = np.delete(sample['true_label'], self.remove_source_ids, axis=0)
        if not condition.any():
            target = np.zeros(shape=[1, *mags.shape[1:]])  # + np.finfo(np.float).eps
        elif np.prod(condition) == 1:
            target = np.sum(mags[:-1], axis=0)[None]
        else:
            selected_id = np.nonzero(condition)[0][0]
            target = mags[selected_id][None]
        input_mix = np.concatenate([target, mags[-1][None]], axis=0)
        return [torch.from_numpy(input_mix).float(), torch.from_numpy(condition).float()], \
               [torch.from_numpy(mixture_phase).unsqueeze(0), file_name, torch.from_numpy(true_label), torch.from_numpy(condition).float()]
