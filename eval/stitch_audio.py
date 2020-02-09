import sys
sys.path.append('../')
from pydub import AudioSegment
from utils.utils import create_folder
from settings import *

SAMPLING_RATE=TARGET_SAMPLING_RATE
dir_path=os.path.join(DUMPS_FOLDER, 'audio', TEST_UNET_CONFIG, 'test')
output_path=os.path.join(DUMPS_FOLDER, 'stitched', TEST_UNET_CONFIG, 'test')
folders=os.listdir(dir_path)
if ISOLATED:
    elements = [SOURCES_SUBSET[ISOLATED_SOURCE_ID]]
    output_path = os.path.join(DUMPS_FOLDER, 'stitched', TYPE+'_baseline', 'test')
else:
    elements = SOURCES_SUBSET
for element in elements:
    for idx, folder in enumerate(folders):
        folder_path = os.path.join(dir_path, folder)
        print('Stitching [{0}/{1}] [TRACK NAME]: {2}'.format(idx, len(folders), folder))
        combined = AudioSegment.empty()
        combined_path = os.path.join(output_path, folder, element+'.wav')
        create_folder(os.path.dirname(combined_path))
        for i in range(len(os.listdir(folder_path))):
            chunk_path = os.path.join(folder_path, str(i), 'PR_'+element+'.wav')
            chunk = AudioSegment.from_wav(chunk_path)
            chunk = chunk.set_frame_rate(SAMPLING_RATE)
            chunk = chunk.set_channels(1)
            combined += chunk
        combined.export(combined_path, format='wav')
