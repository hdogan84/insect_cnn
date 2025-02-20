import os
from pathlib import Path
import sys
import shutil
import glob

# env to use: clef

user_root = Path().resolve().parent.parent
sys.path.append(user_root)

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
        print(f"Moved: {file_path} -> {destination_folder}")

target_class = 0

audio_folder = str(user_root) + '/Woodcock-CNN/data/train_data/audio/' + str(target_class)
embedding_folder = str(user_root) + '/Woodcock-CNN/data/train_data/embedding/pyAudio/' + str(target_class)

mT.mid_feature_extraction_file_dir(audio_folder, 3.0, 1.0, 0.1, 0.1)

## move npy files to embedding dir
move_npy_files(audio_folder, embedding_folder)
