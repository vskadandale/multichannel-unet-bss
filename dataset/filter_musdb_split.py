import sys
sys.path.append('..')
import numpy as np
from settings import *

folders_path=os.path.join(CHUNKS_PATH,'train')
folders=os.listdir(folders_path)
selected_files=[]
for folder in folders:
    sub_folders_path=os.path.join(folders_path,folder)
    sub_folders=os.listdir(sub_folders_path)
    for sub_folder in sub_folders:
        samples_path=os.path.join(sub_folders_path,sub_folder)
        samples=os.listdir(samples_path)
        select_sample = True
        for sample in samples:
            cla,energy=sample[:-4].split('_')
            if (cla in SOURCES_SUBSET) and (energy=='0'):
                select_sample=False
                break
        if select_sample:
            selected_files.append(os.path.join(MUSDB_SPLITS_PATH,'train',folder,sub_folder)+'.npy')

np.save(FILTERED_SAMPLE_PATHS,selected_files)

