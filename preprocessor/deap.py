import os
import csv
import numpy as np
import pandas as pd
import youtube_dl
import pickle
import multiprocessing
from collections import Counter
from functools import partial
from contextlib import contextmanager
import pyeeg as pe
import torch
from tqdm import tqdm

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

def fft_preprocess(data, channel, band, window_size, sample_rate, step_size):
    start = 0
    meta_array = []
    while start + window_size < data.shape[1]:
        meta_data = [] #meta vector for analysis
        for j in channel:
            X = data[j][start : start + window_size] #Slice raw data over 2 sec, at interval of 0.125 sec
            Y = pe.bin_power(X, band, sample_rate) #FFT over 2 sec of channel j, in seq of theta, alpha, low beta, high beta, gamma
            meta_data.extend(Y[0])
        meta_array.append(np.array(meta_data))
        start = start + step_size
    return np.array(meta_array)

def eeg_processor(path):
    subjectList = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32']
    channel = [1,2,3,4,6,11,13,17,19,20,21,25,29,31] #14 Channels chosen to fit Emotiv Epoch+
    band = [4,8,12,16,25,45] #5 bands
    window_size = 256 #Averaging band power of 2 sec
    step_size = 16 #Each 0.125 sec update once
    sample_rate = 128 #Sampling rate of 128 Hz
    dirs = os.path.join(path, "data_preprocessed_python/")
    final_annotation = {}
    for sub in tqdm(subjectList):
        with open(f"{dirs}s{sub}.dat", 'rb') as file:
            subject = pickle.load(file, encoding='latin1') #resolve the python 2 data problem by encoding : latin1
            for trial in range (0,40):
                # loop over 0-39 trails
                data = subject["data"][trial]
                label = subject["labels"][trial]
                feature = fft_preprocess(data, channel, band, window_size, sample_rate, step_size)
                final_annotation[f"{sub}_{trial}"] = {
                    "subject": sub,
                    "trial": trial,
                    "data": data,
                    "label": label,
                    "feature": feature
                }
    torch.save(final_annotation, os.path.join(path, "annotation.pt"))

def DEAP_preprocssor(path):
    df = pd.read_csv(os.path.join(path, "video_list.csv"))
    paths = [path for _ in range(len(df))]
    _ids = list(df['Online_id'])
    urls = list(df['Youtube_link'])
    with poolcontext(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(audio_crawl, zip(_ids, urls, paths))
    print("finish extract")
    eeg_processor(path)