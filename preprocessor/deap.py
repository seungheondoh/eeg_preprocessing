import os
import csv
import librosa
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
from .constants import BAND, DEAP_Start, LABELS, DEAP_CHANNEL, LOC2D, MUSIC_SAMPLE_RATE
from .eeg_utils import get_psd, psd_data, deap_label, dataset_to_img_feature, gen_images

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
    """
    params
        path: 원본 데이터 위치
        info: mne information

    return -> None
    save -> raw_annotation, stft_annotation, psd_annotation, img_annotation -> Dict 
        key : sub-피험자 + idx-video
        value -> Dict
            feature: Channel x Singal or Channel x Freq x Time
    """
    subjectList = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32']
    channel = [i for i in range(32)] #14 Channels chosen to fit Emotiv Epoch+
    participant_ratings = pd.read_csv(os.path.join(path, 'participant_ratings.csv'))
    wav_dict = {}
    for experiment_id in set(participant_ratings['Experiment_id']):
        y, sr = librosa.load(os.path.join(path,"wav",str(experiment_id)+".mp3"), sr=MUSIC_SAMPLE_RATE, res_type='kaiser_fast')
        wav_dict[experiment_id] = y
    dirs = os.path.join(path, "data_preprocessed_python/")
    raw_annotation, stft_annotation, psd_annotation, img_annotation = {}, {}, {}, {}
    img_features = []
    for sub in tqdm(subjectList):
        with open(f"{dirs}s{sub}.dat", 'rb') as file:
            subject = pickle.load(file, encoding='latin1') #resolve the python 2 data problem by encoding : latin1
        df_paticipant = participant_ratings[participant_ratings['Participant_id'] == int(sub)]
        for idx in range (0,40):
            experiment_id = idx + 1 # matlab index
            y = wav_dict[experiment_id]
            start_time = df_paticipant[df_paticipant['Experiment_id'] == experiment_id]['Start_time'].values[0]
            start_idx = int((start_time / 1e6) * MUSIC_SAMPLE_RATE)
            wav = y[start_idx: start_idx+(MUSIC_SAMPLE_RATE*60)]
            # load eeg
            data = subject["data"][idx]
            data = data[:32, DEAP_Start:]
            label = subject["labels"][idx]
            stft = mne.time_frequency.stft(data, wsize=128)
            av_label, a_label, v_label = deap_label(label)
            raw = mne.io.RawArray(data, info)
            psd_feature = psd_data(raw)
            image_feature = dataset_to_img_feature(data)
            img_features.append(image_feature)

            raw_annotation[f"{sub}_{idx}"] = {
                "feature": data,
                "av_label": av_label,
                "a_label": a_label,
                "v_label": v_label,
                "label": label,
                "wav": wav
            }

            stft_annotation[f"{sub}_{idx}"] = {
                "feature": np.abs(stft).astype(np.float32),
                "av_label": av_label,
                "a_label": a_label,
                "v_label": v_label,
                "label": label,
                "wav": wav
            }

            psd_annotation[f"{sub}_{idx}"] = {
                "feature": psd_feature,
                "av_label": av_label,
                "a_label": a_label,
                "v_label": v_label,
                "label": label,
                "wav": wav
            }

            img_annotation[f"{sub}_{idx}"] = {
                "feature": image_feature,
                "av_label": av_label,
                "a_label": a_label,
                "v_label": v_label,
                "label": label,
                "wav": wav
            }
    img_features = np.array(img_features)
    images = torch.tensor([gen_images(np.array(LOC2D),
                        img_features[:, i * 192:(i + 1) * 192], 32, normalize=True) for i in range(int(img_features.shape[1] / 192))
               ])
    torch.save(raw_annotation, os.path.join(path, "raw_data.pt"))
    torch.save(stft_annotation, os.path.join(path, "stft_data.pt"))
    torch.save(psd_annotation, os.path.join(path, "psd_data.pt"))
    # torch.save(img_annotation, os.path.join(path, "img_data.pt"))
    # torch.save(images, os.path.join(path, "image_features.pt"))

def DEAP_preprocssor(path):
    # df = pd.read_csv(os.path.join(path, "video_list_fixed.csv"))
    # paths = [path for _ in range(len(df))]
    # _ids = list(df['Online_id'])
    # urls = list(df['Youtube_link'])
    # with poolcontext(processes=10) as pool:
    #     pool.starmap(audio_crawl, zip(_ids, urls, paths))
    # print("finish extract")
    info = mne.create_info(32, sfreq=128)
    info = mne.create_info(DEAP_CHANNEL, ch_types=32*['eeg'], sfreq=128)
    eeg_processor(path, info)