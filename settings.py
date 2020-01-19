import os

def set_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

TYPE='4src'#'4src'
ISOLATED_SOURCE_ID=0
SOURCES=['vocals', 'accompaniment', 'drums', 'bass', 'other']
if TYPE=='2src':
    SOURCES_SUBSET=['vocals', 'accompaniment']
else:
    SOURCES_SUBSET=['vocals', 'drums', 'bass', 'other']

TARGET_SAMPLING_RATE=10880
DURATION=6                                                      #in seconds
NFFT=1022
HOP_LENGTH=256
STFT_WIDTH=int((TARGET_SAMPLING_RATE * DURATION / HOP_LENGTH) + 1)   # 256=(10880x6/256)+1

K = len(SOURCES_SUBSET)  # Number of instruments
#self.rootdir = set_path(EXPERIMENTS_FOLDER)
BATCH_SIZE = 16
LR = 0.001                          # \\TODO 0.01
EPOCHS = 60000
DWA_TEMP = 2
MOMENTUM = 0                        # \\TODO 0.9
WEIGHT_DECAY = 0
INITIALIZER = 'xavier'
OPTIMIZER = 'SGD'
USE_BN = True
PRETRAINED=None                                                 ### None or True ???????????????
#self.datadir = MUSDB_SPLITS_PATH
TRACKGRAD=False
ACTIVATION = None
INPUT_CHANNELS = 1

###### WEIGHTS #######
PRETRAINED_UNET_CONFIG='2019-12-01 15:01:42'

#### TENSORBOARD CONFIG #####
PARAMETER_SAVE_FREQUENCY=100


"""
MUSDB_FOLDER_PATH='/media/venkatesh/slave/dataset/musdb'
INDIAN_SAMPLE_DATA='/media/venkatesh/slave/dataset/Indian_Music/sample/X'
EXPERIMENTS_FOLDER='/media/venkatesh/slave/weights'
DUMPS_FOLDER='/media/venkatesh/slave/dumps'
SEPERATED_OUTPUT_PATH='/media/venkatesh/slave/dataset/output_crumbs/'
OUTPUT_PATH='/media/venkatesh/slave/dataset/output'
RESULTS_PATH='/media/venkatesh/slave/dataset/results'
"""

MUSDB_FOLDER_PATH='/mnt/DATA/datasets/musdb/'
INDIAN_SAMPLE_DATA='/mnt/DATA/datasets/Indian_Music/sample/X'
EXPERIMENTS_FOLDER='/mnt/DATA/weights'
DUMPS_FOLDER='/mnt/DATA/dumps'
SEPERATED_OUTPUT_PATH='/mnt/DATA/datasets/musdb/output_crumbs'
OUTPUT_PATH='/mnt/DATA/datasets/musdb/output'
RESULTS_PATH='/mnt/DATA/datasets/musdb/results'




ROOT_DIR = set_path(EXPERIMENTS_FOLDER)
PRETRAINED_UNET_WEIGHTS_PATH=os.path.join(EXPERIMENTS_FOLDER,PRETRAINED_UNET_CONFIG,'bestcheckpoint.pth')
TEST_UNET_CONFIG='2020-01-17 14:15:31'#'2020-01-03 11:42:35'#'2020-01-02 19:19:54'#'baseline'#'2020-01-01 20:03:30'#'2019-12-31 14:27:24'#'2019-12-18 18:53:17'
TEST_UNET_WEIGHTS_PATH=os.path.join(EXPERIMENTS_FOLDER,TEST_UNET_CONFIG,'bestcheckpoint.pth')
TEST_UNET_REFINED_CONFIG='2019-12-18 18:53:17'
TEST_UNET_REFINED_WEIGHTS_PATH=os.path.join(EXPERIMENTS_FOLDER,TEST_UNET_REFINED_CONFIG,'bestcheckpoint.pth')
RAW_MUSDB_PATH=os.path.join(MUSDB_FOLDER_PATH,'musdb18')
MUSDB_WAVS_FOLDER_PATH=os.path.join(MUSDB_FOLDER_PATH,'musdb18_wavs')
ENERGY_PROFILE_FOLDER=os.path.join(MUSDB_FOLDER_PATH,'energy_profile')
MUSDB_SPLITS_PATH=os.path.join(MUSDB_FOLDER_PATH,'musdbsplit')
MUSDB_SPLITS_AUG_PATH=os.path.join(MUSDB_FOLDER_PATH,'musdbaug')
CHUNKS_PATH=os.path.join(MUSDB_FOLDER_PATH,'musdb_chunks')
SPECTROGRAMS_PATH=os.path.join(MUSDB_FOLDER_PATH,'musdb_spectrograms')
SOURCE_MIX_PATH=os.path.join(MUSDB_FOLDER_PATH,'musdb_smix')
STEMS_PATH=os.path.join(MUSDB_FOLDER_PATH,'musdb_stems')
RECON_GT_PATH=os.path.join(MUSDB_FOLDER_PATH,'musdbGT')
SOURCE_ESTIMATES_PATH=os.path.join(MUSDB_FOLDER_PATH,'eval')
TEST_DATA_PATH=os.path.join(RAW_MUSDB_PATH,'test')
TEST_SPEC_DATA_PATH=os.path.join(MUSDB_SPLITS_PATH,'test')
TEST_MAPPINGS_PATH=os.path.join(os.path.dirname(SEPERATED_OUTPUT_PATH),'test_mappings.npy')

SOURCES_SUBSET_ID=[SOURCES.index(i) for i in SOURCES_SUBSET]
SAVE_SEGMENTS=True
ENERGY_THRESHOLD=0

FILTERED_SAMPLE_PATHS=os.path.join(MUSDB_FOLDER_PATH,TYPE+'_filtered')


