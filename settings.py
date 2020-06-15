import os


def set_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


MAIN_DEVICE = 0                       #Selects the GPU by its id
TYPE = '2src'  # '4src'               #Set to either '2src' or '4src' to choose between singing voice separation mode and multi-instrument separation mode
ISOLATED = False                      #Set to True only when running baseline.py otherwise, set to False 
ISOLATED_SOURCE_ID = 0                #When ISOLATED==True, this setting chooses which dedicated source to select based on the id provided here.
SOURCES = ['vocals', 'accompaniment', 'drums', 'bass', 'other']
if TYPE == '2src':
    SOURCES_SUBSET = ['vocals', 'accompaniment']
else:
    SOURCES_SUBSET = ['vocals', 'drums', 'bass', 'other']

ORIGINAL_SAMPLING_RATE = 44100
TARGET_SAMPLING_RATE = 10880         #Set the downsampling rate.
DURATION = 6  # in seconds           #Set the duration of sample excerpt   
NFFT = 1022                          #Set the NFFT parameter for STFT
HOP_LENGTH = 256                     #Set the Hop Length parameter for STFT 
STFT_WIDTH = int((TARGET_SAMPLING_RATE * DURATION / HOP_LENGTH) + 1)  # 256=(10880x6/256)+1

K = len(SOURCES_SUBSET)              #Number of instruments
BATCH_SIZE = 16                      #Set the batch size
LR = 0.01                            #Set the learning rate  
EPOCHS = 60000  # 500                #Set the maximum number of epochs
DWA_TEMP = 2                         #Set the temperature for DWA (only relevant for DWA experiments) 
MOMENTUM = 0.9                       #Set the optimizer momentum
DROPOUT = 0.1                        #Set the dropout
WEIGHT_DECAY = 0                     #Set the weight decay
INITIALIZER = 'xavier'               #Set the optimizer initializer
OPTIMIZER = 'SGD'                    #Set the optimizer type
USE_BN = True                        #Set True for Batch Normalization           
PRETRAINED = None
TRACKGRAD = False
ACTIVATION = None
INPUT_CHANNELS = 1                   #Number of input channels to the model
EARLY_STOPPING_PATIENCE = 60         #Set the early stopping patience

# CUNet Settings
FILTERS_LAYER_1 = 32
Z_DIM = 4
N_CONDITIONS = 4064
N_NEURONS = [32, 512, 4096]
CUNET_DROPOUT = 0.1

##### ENERGY STATS #####
ACC_ENERGY = 687.5261
BAS_ENERGY = 252.7046
DRU_ENERGY = 218.6938
MIX_ENERGY = 858.8005
OTH_ENERGY = 216.4932
VOC_ENERGY = 173.4346

#### SPECTROGRAM CHANNEL U-NET ####
if TYPE == '2src':
    w_1 = 0.7986   # w_V x VOC_ENERGY = (1-w_V) x ACC_ENERGY, w_V = ACC_ENERGY/(ACC_ENERGY + VOC_ENERGY)
    w_2 = 1 - w_1  # w_V = 687.5261/(687.5261 + 173.4346)
else:
    w_1 = 0.3048   # w_V*V_NRG = w_D*D_NRG = w_B*B_NRG = w_O*O_ENERGY
    w_2 = 0.2417   # w_V + w_D + w_B + w_O = 1
    w_3 = 0.2092   # w_V*173.4346 = w_D*218.6938 = w_B*252.7046 = w_O*216.4932
    w_4 = 0.2442

#### TENSORBOARD CONFIG #####
PARAMETER_SAVE_FREQUENCY = 100           #Set the parameter save frequency for tensorboard

##### Main Directory Path #####
#MAIN_DIR_PATH = '/media/venkatesh/slave'
MAIN_DIR_PATH = '/mnt/DATA'

#Set the model id for testing
TEST_UNET_CONFIG = '2020-02-10 14:55:38' #'2020-06-12 20:52:47'   # '2020-06-12 20:54:17'

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
ENERGY_THRESHOLD = 0                    #Set the energy threshold for considering a sample as silent.

FILTERED_SAMPLE_PATHS = os.path.join(MUSDB_FOLDER_PATH, TYPE + '_filtered')
