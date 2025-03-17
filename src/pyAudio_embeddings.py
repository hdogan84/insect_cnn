import os
from pathlib import Path
import sys
import shutil
import glob

# env to use: clef

user_root = Path().resolve().parent.parent
sys.path.append(user_root)
data_dir = str(user_root) + '/Woodcock-CNN/data/'

pyAudio_path = user_root / "pyAudioAnalysis"
pyAudio_pth = os.path.abspath(pyAudio_path)
sys.path.append(pyAudio_pth)
sys.path.append(pyAudio_path)

from pyAudioAnalysis import MidTermFeatures as mT 
import subprocess


def move_npy_files(source_folder, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)  # Ensure destination exists
    for file_path in glob.glob(os.path.join(source_folder, "*.npy")):
        shutil.move(file_path, os.path.join(destination_folder, os.path.basename(file_path)))
        #print(f"Moved: {file_path} -> {destination_folder}")

target_class = 1
mt_step = 3

audio_folder = data_dir + 'train_data/audio/' + str(target_class)
embedding_folder = data_dir + 'train_data/embedding/pyAudio/' + str(mt_step) + 's_step/' + str(target_class)

mT.mid_feature_extraction_file_dir(audio_folder, 3.0, 3.0, 0.2, 0.2)

print("Completed feature extraction")

## move npy files to embedding dir
move_npy_files(audio_folder, embedding_folder)

print("Completed moving npy files")