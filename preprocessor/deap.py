import os
import csv
import torch
import numpy as np
import pandas as pd
import youtube_dl
import pickle
import multiprocessing
from collections import Counter
from functools import partial
from contextlib import contextmanager
from tqdm import tqdm
from sklearn import preprocessing
import mne
from .constants import BAND, DEAP_Start, LABELS, DEAP_CHANNEL
from .eeg_utils import get_psd, psd_data, deap_label_encoder

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    

def audio_crawl(_id, url, path):
    audio_out_dir = os.path.join(path, 'wav', str(_id) + ".")
    error_dir = os.path.join(path, 'error', str(_id) + ".npy")
    ydl_opts = {
        'format': 'bestaudio/best',
        'extractaudio' : True,      # only keep the audio
        'audioformat' : 'mp3',      # convert to mp3
        'writeinfojson': False,
        'noplaylist': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'postprocessor_args': [
            '-ar', '22050'
        ],
        'outtmpl': audio_out_dir
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download = True)
    except:
        np.save(os.path.join(error_dir), _id)

def eeg_processor(path, info):
    lb = preprocessing.LabelBinarizer()
    lb.fit(LABELS)

    subjectList = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32']
    channel = [i for i in range(32)] #14 Channels chosen to fit Emotiv Epoch+
    dirs = os.path.join(path, "data_preprocessed_python/")
    final_annotation = {}
    for sub in tqdm(subjectList):
        with open(f"{dirs}s{sub}.dat", 'rb') as file:
            subject = pickle.load(file, encoding='latin1') #resolve the python 2 data problem by encoding : latin1
            for trial in range (0,40):
                # loop over 0-39 trails
                data = subject["data"][trial]
                data = data[:32, DEAP_Start:]
                label = subject["labels"][trial]

                stft = mne.time_frequency.stft(data, wsize=128)
                cls_label, binary = deap_label_encoder(label, lb)
                raw = mne.io.RawArray(data, info)
                psd_feature = psd_data(raw)

                final_annotation[f"{sub}_{trial}"] = {
                    "subject": sub,
                    "trial": trial,
                    "data": data,
                    "stft": stft,
                    "cls_label": cls_label,
                    "binary": binary,
                    "psd_feature": psd_feature,
                    "label": label,
                }
    torch.save(final_annotation, os.path.join(path, "annotation.pt"))

def DEAP_preprocssor(path):
    # df = pd.read_csv(os.path.join(path, "video_list.csv"))
    # paths = [path for _ in range(len(df))]
    # _ids = list(df['Online_id'])
    # urls = list(df['Youtube_link'])
    # with poolcontext(processes=multiprocessing.cpu_count()) as pool:
    #     pool.starmap(audio_crawl, zip(_ids, urls, paths))
    # print("finish extract")
    info = mne.create_info(32, sfreq=128)
    info = mne.create_info(DEAP_CHANNEL, ch_types=32*['eeg'], sfreq=128)
    eeg_processor(path, info)