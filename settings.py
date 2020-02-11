import os


def set_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


MAIN_DEVICE = 1
TYPE = '2src'  # '4src'
ISOLATED = False
ISOLATED_SOURCE_ID = 0
SOURCES = ['vocals', 'accompaniment', 'drums', 'bass', 'other']
if TYPE == '2src':
    SOURCES_SUBSET = ['vocals', 'accompaniment']
else:
    SOURCES_SUBSET = ['vocals', 'drums', 'bass', 'other']

ORIGINAL_SAMPLING_RATE = 44100
TARGET_SAMPLING_RATE = 10880
DURATION = 6  # in seconds
NFFT = 1022
HOP_LENGTH = 256
STFT_WIDTH = int((TARGET_SAMPLING_RATE * DURATION / HOP_LENGTH) + 1)  # 256=(10880x6/256)+1

K = len(SOURCES_SUBSET)  # Number of instruments
BATCH_SIZE = 16
LR = 0.01
EPOCHS = 60000  # 500
DWA_TEMP = 2
MOMENTUM = 0.9
DROPOUT = 0.1
WEIGHT_DECAY = 0
INITIALIZER = 'xavier'
OPTIMIZER = 'SGD'
USE_BN = True
PRETRAINED = None
TRACKGRAD = False
ACTIVATION = None
INPUT_CHANNELS = 1
EARLY_STOPPING_PATIENCE = 60

# CUNet Settings
FILTERS_LAYER_1 = 16
CONTROL_TYPE = 'dense'
FILM_TYPE = 'complex'
Z_DIM = 4
N_CONDITIONS = 1008  # 4064
N_NEURONS = [16, 128, 1024]

##### ENERGY STATS #####
ACC_ENERGY = 687.5261
BAS_ENERGY = 252.7046
DRU_ENERGY = 218.6938
MIX_ENERGY = 858.8005
OTH_ENERGY = 216.4932
VOC_ENERGY = 173.4346

#### TENSORBOARD CONFIG #####
PARAMETER_SAVE_FREQUENCY = 100

##### Main Directory Path #####
#MAIN_DIR_PATH = '/media/venkatesh/slave'
MAIN_DIR_PATH = '/mnt/DATA'
#MAIN_DIR_PATH = '/homedtic/vshenoykadandale'


TEST_UNET_CONFIG = '2020-02-11 13:00:57'  # '2020-01-03 11:42:35'#'2020-01-02 19:19:54'#'baseline'#'2020-01-01 20:03:30'#'2019-12-31 14:27:24'#'2019-12-18 18:53:17'

MUSDB_FOLDER_PATH = os.path.join(MAIN_DIR_PATH, 'dataset', 'musdb')
EXPERIMENTS_FOLDER = os.path.join(MAIN_DIR_PATH, 'weights')
DUMPS_FOLDER = os.path.join(MAIN_DIR_PATH, 'dumps')
ROOT_DIR = set_path(EXPERIMENTS_FOLDER)
TEST_UNET_WEIGHTS_PATH = os.path.join(EXPERIMENTS_FOLDER, TEST_UNET_CONFIG, 'bestcheckpoint.pth')
RAW_MUSDB_PATH = os.path.join(MUSDB_FOLDER_PATH, 'musdb18')
MUSDB_WAVS_FOLDER_PATH = os.path.join(MUSDB_FOLDER_PATH, 'musdb18_wavs')
ENERGY_PROFILE_FOLDER = os.path.join(MUSDB_FOLDER_PATH, 'energy_profile')
MUSDB_SPLITS_PATH = os.path.join(MUSDB_FOLDER_PATH, 'musdbsplit')
CHUNKS_PATH = os.path.join(MUSDB_FOLDER_PATH, 'musdb_chunks')

SOURCES_SUBSET_ID = [SOURCES.index(i) for i in SOURCES_SUBSET]
ENERGY_THRESHOLD = 0

FILTERED_SAMPLE_PATHS = os.path.join(MUSDB_FOLDER_PATH, TYPE + '_filtered')
