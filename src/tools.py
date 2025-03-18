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


def butter_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def extract_audio(sig, rate, begin_times, base_folder, main_wav, length):

    for t_begin in begin_times:
        t_begin = int(t_begin)
        index_begin=t_begin*rate
        # Take  segment length of 2 seconds
        index_end=index_begin+length*rate
        segment=sig[index_begin:index_end]
        #print(len(segment))
        filename=base_folder+main_wav[:-4]+"_"+str(t_begin)+'.wav'
        #print(filename)
        sf.write(filename, segment, rate)



def audio_to_image(audio_path, Nfft, N_overlap, resize_tuple):
    (sig, rate) = librosa.load(audio_path, sr=None, mono=True)
    f, t, Syy = signal.stft(sig,fs=rate,window='hann',noverlap=N_overlap,nfft=Nfft,nperseg=Nfft)
    Syy=abs(Syy) * 33 #apply amplification
    Syy=20*np.log10(Syy/0.00002)
    Syy[Syy<0.0] = 0.0
    Syy = Syy / Syy.max() #normalize to [0, 1]
    Syy *= 255

    Syy_3d = np.stack([Syy] * 3, axis=-1)
    
    Syy_3d = cv2.flip(Syy_3d, 0)
    image = cv2.resize(Syy_3d, resize_tuple)

    return image


def get_embedding_birdnet(root_folder, pos_or_neg):
    
    features = []
    embed_folder = root_folder+str(pos_or_neg)+'/'
    embeddings_txt_files = os.listdir(embed_folder)

    for filename in embeddings_txt_files:
        txt_path = Path(os.path.join(embed_folder, filename))

        with open(txt_path, "r") as file:
            values = file.readlines()[0].split("\t")[2]
            float_vals = [float(x) for x in values.split(",")]
        features.append(float_vals)

    return features


def get_embedding_pyAudio(root_folder, pos_or_neg):
    
    features = []

    folder_path = root_folder+str(pos_or_neg)+'/'
    npy_files = glob.glob(f"{folder_path}/*.npy")
    #embeddings_npy_files = os.listdir(embed_folder)

    for npy_path in npy_files:

        arr= np.load(npy_path).reshape(-1)
        features.append(arr)

    return features


