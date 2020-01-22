import sys

sys.path.append('../')
from settings import *
import pandas as pd
import numpy as np
from utils import create_folder
import librosa
import mir_eval

SAMPLING_RATE = TARGET_SAMPLING_RATE
dir_path = os.path.join(DUMPS_FOLDER, 'audio', TEST_UNET_CONFIG, 'test')
output_path = os.path.join(DUMPS_FOLDER, 'stitched', TEST_UNET_CONFIG, 'test')
folders = os.listdir(output_path)

metadata = ['filename', *[y + x for y in ['SDR_', 'SIR_', 'SAR_'] for x in SOURCES_SUBSET]]
df = pd.DataFrame(columns=metadata)
category = 'test'
setting = TEST_UNET_CONFIG
sr = TARGET_SAMPLING_RATE
GT = os.path.join(MUSDB_WAVS_FOLDER_PATH + '_' + str(TARGET_SAMPLING_RATE))
COMPARISON = os.path.join(DUMPS_FOLDER, 'stitched',
                          TEST_UNET_CONFIG)  # [resampled_output_path,phasemixed_output_path,resampled_phasemixed_ouput_path]
results_folder = os.path.join(DUMPS_FOLDER, 'results',
                              setting)  # {1.ori:mixphased,resampled,resampled_mixphased; 2.down:mixphased_10800} ##TODO stereo
create_folder(results_folder)

folders = sorted(os.listdir(os.path.join(GT, category)))
for i, folder in enumerate(folders):
    print('[{0}/{1}] [TRACK NAME]: {2}'.format(i, len(folders), folder))
    for idx, source in enumerate(SOURCES_SUBSET):
        gt_i, _ = librosa.load(os.path.join(GT, category, folder, source + '.wav'), sr=sr)
        y_i, _ = librosa.load(os.path.join(COMPARISON, category, folder, source + '.wav'), sr=sr)

        if idx == 0:
            L = len(gt_i)
            gt = np.zeros([len(SOURCES_SUBSET), L])
            y = np.zeros([len(SOURCES_SUBSET), L])

        gt[idx] = gt_i[:L]
        del gt_i
        y[idx] = y_i[:L]
        del y_i
        # gt = gt[:,:y.shape[0]] ##Also measure the impact of this. notice that min and max value indices of gt and estimates do not always coincide (but are closeby)

    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(gt, y)
    row = [*sdr, *sir, *sar]
    print('Perm : ' + str(perm))
    del sdr, sar, sir, perm
    df.loc[i] = [folder, *row]
    del row

pd.DataFrame.to_csv(df, path_or_buf=os.path.join(results_folder, category + '_metrics.csv'), index=False)
