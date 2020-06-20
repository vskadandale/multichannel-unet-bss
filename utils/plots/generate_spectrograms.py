import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import librosa
import librosa.display
from settings import *

"""
path = os.path.join(DUMPS_FOLDER, 'visuals', TEST_UNET_CONFIG, 'test')
folders = os.listdir(path)
for folder in folders:
    folder_path = os.path.join(path, folder)
    sub_folders = os.listdir(folder_path)
    for sub_folder in sub_folders:
        image_path = os.path.join(folder_path,sub_folder,'vocals_MAG_GT.png')
        img = mpimg.imread(image_path)
        plt.figure()
        plt.imshow(img)
        plt.show()
        plt.close()

path = os.path.join(DUMPS_FOLDER, 'MMSP2020_selected', TYPE)
folders = os.listdir(path)
for folder in folders:
    folder_path = os.path.join(path, folder,'audio')
    visual_path = os.path.join(path, folder,'visuals')
    sub_folders = os.listdir(folder_path)
    for sub_folder in sub_folders:
        sub_folder_path = os.path.join(folder_path,sub_folder)
        sub_sub_folders = os.listdir(sub_folder_path)
        for sub_sub_folder in sub_sub_folders:
            sub_sub_folder_path = os.path.join(sub_folder_path,sub_sub_folder)
            files = os.listdir(sub_sub_folder_path)
            for file in files:
                file_path = os.path.join(sub_sub_folder_path,file)
                output_path = os.path.join(visual_path,sub_folder,sub_sub_folder)
                set_path(output_path)
                y, sr = librosa.load(file_path,sr=None)
                stft = librosa.stft(y, n_fft=NFFT, hop_length=HOP_LENGTH)
                D = librosa.amplitude_to_db(np.absolute(stft),
                                            ref=np.max)
                librosa.display.specshow(D, x_axis='time', sr=TARGET_SAMPLING_RATE,cmap='Reds', y_axis='linear')
                plt.ylabel(None)
                plt.xlabel(None)
                plt.savefig(os.path.join(output_path,file[:-4]+'.png'), bbox_inches='tight')
                #plt.show()
"""
dir_path = os.path.join(DUMPS_FOLDER, 'MMSP2020_selected', '4src')
song = 'The Doppler Shift - Atrophy'
model='du'
paths=[os.path.join(dir_path, model, 'audio',song,'39')
       ]
count=39
for path in paths:
    files = os.listdir(path)
    #count+=1
    for file in files:
        file_path = os.path.join(path, file)
        output_path = os.path.join(dir_path, model,'visuals',song, str(count))
        set_path(output_path)
        y, sr = librosa.load(file_path, sr=None)
        stft = librosa.stft(y, n_fft=NFFT, hop_length=HOP_LENGTH)
        D = librosa.amplitude_to_db(np.absolute(stft),
                                    ref=np.max)
        librosa.display.specshow(D, x_axis='time', sr=TARGET_SAMPLING_RATE, cmap='Reds', y_axis='linear')
        plt.ylabel(None)
        plt.xlabel(None)
        plt.savefig(os.path.join(output_path, file[:-4] + '.png'), bbox_inches='tight')
        # plt.show()
