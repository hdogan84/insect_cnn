import os
from pathlib import Path
import sys

user_root = Path().resolve().parent.parent
sys.path.append(user_root)

birdnet_path = user_root / "BirdNET_Analyzer"
birdnet_pth = os.path.abspath(birdnet_path)
sys.path.append(birdnet_pth)
sys.path.append(birdnet_path)

import birdnet_analyzer
import subprocess


audio_folder = str(user_root) + '/Woodcock-CNN/data/train_data/audio/1/'
embedding_folder = str(user_root) + '/Woodcock-CNN/data/train_data/embedding/birdnet/1'


subprocess.run(
    ["python", "-m", "birdnet_analyzer.embeddings", "--i", audio_folder, "--o", embedding_folder],
    cwd=str(birdnet_path)
)

