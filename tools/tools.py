import librosa
from scipy.signal import butter, sosfilt, lfilter
from scipy import signal
from PIL import Image
import numpy as np
import cv2
import os, glob
from pathlib import Path
import soundfile as sf
import random
import string

def generate_filename():
    filename = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    return filename + '.***'














