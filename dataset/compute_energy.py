import numpy as np
import pandas as pd
import csv
from settings import *

chunks_path=os.path.join(CHUNKS_PATH, 'train')
tracks = os.listdir(chunks_path)
trackwise_energy = pd.DataFrame(columns=['Name','Vocals','Accompaniment','Drums','Bass','Other'])
for id, track in enumerate(sorted(tracks)):
    track_path = os.path.join(chunks_path, track)
    sub_tracks = os.listdir(track_path)
    current_energy = np.zeros(5)
    num_sub = len(sub_tracks)
    v, a, d, b, o = 0, 0, 0, 0, 0
    for sub_track in sub_tracks:
        sub_track_path = os.path.join(track_path, sub_track)
        filenames = os.listdir(sub_track_path)
        for filename in filenames:
            if filename.__contains__('MIX'):
                continue
            elif filename.__contains__('vocals'):
                v += int(filename[:-4].split('_')[-1])
                continue
            elif filename.__contains__('accompaniment'):
                a += int(filename[:-4].split('_')[-1])
                continue
            elif filename.__contains__('drums'):
                d += int(filename[:-4].split('_')[-1])
                continue
            elif filename.__contains__('bass'):
                b += int(filename[:-4].split('_')[-1])
                continue
            elif filename.__contains__('other'):
                o += int(filename[:-4].split('_')[-1])
                continue
    current_energy = np.array([v, a, d, b, o])/num_sub
    trackwise_energy.loc[id] = np.array([track, *current_energy])

trackwise_energy.to_csv(os.path.join(ENERGY_PROFILE_FOLDER, 'trackwise_energy_profile.csv'), index=False, header=True)
