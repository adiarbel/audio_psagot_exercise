import random
import torch
from torch.utils.data import Sampler
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from torch.utils.data import Dataset
from python_speech_features import mfcc
import os
import scipy
import numpy as np
import warnings
warnings.filterwarnings("ignore")

base_dir = '../feature_engineering_Adi/full_dataset'

def pad_batch(batch):
    mfccs, labels = zip(*batch)
    mfccs_padded = pad_sequence([torch.tensor(b_item, dtype=torch.float) for b_item in mfccs], batch_first=True, padding_value=0)
    
    return mfccs_padded, torch.tensor(labels, dtype=torch.long)
    
def get_wav_path(fname, train=True):
    if train:
        return os.path.join(base_dir,'audio_train',fname)
    else:
        return os.path.join(base_dir,'audio_test',fname)

def duration_func(wav_loc, train=True):
    return torchaudio.info(get_wav_path(wav_loc, train)).num_frames

class MFCCDataset(Dataset):
    def __init__(self, df, train=True):
        self.train = train
        self.df = df
        self.df['duration'] = df.index.map(lambda x: duration_func(x, train))
        self.df.sort_values('duration',inplace=True)
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        line = self.df.iloc[idx]
        rate, signal = scipy.io.wavfile.read(get_wav_path(self.df.index[idx], self.train))
        fixed_length = 10*rate
        signal = signal[:fixed_length]
        if len(signal) < fixed_length:
            signal = np.pad(signal, (0,fixed_length-len(signal)), constant_values=0)
        mfcc_res = mfcc(signal, rate, numcep=13, nfilt=26, nfft=1103)

        return mfcc_res, line.label_id

# class STTSampler(Sampler):
#     def __init__(self, dataset: STTDataset, 
#                  max_duration_per_batch: float,
#                  max_audio_duration: float,
#                  min_audio_duration: float,
#                  max_batch_size: int):
        
#         self.max_audio_duration = max_audio_duration
#         self.min_audio_duration = min_audio_duration
#         self.max_duration_per_batch = max_duration_per_batch
#         self.max_batch_size = max_batch_size
#         self.dataset = dataset
#         self.batches_list = []
        
#         batch = []
#         batch_audio_durations = []
        
#         for i, stm_line in enumerate(self.dataset.stm_lines):
#             duration = stm_line.duration
#             if duration > self.max_audio_duration or duration < self.min_audio_duration:
#                 continue
            
#             if self._calc_padded_batch_length(batch_audio_durations+[duration]) <= self.max_duration_per_batch and len(batch) <= self.max_batch_size-1:
#                 batch.append(i) 
#                 batch_audio_durations.append(duration)
#             else:
#                 self.batches_list.append(batch)
#                 batch = [i]
#                 batch_audio_durations = [duration]
                
#         self.batches_list.append(batch)
#         random.shuffle(self.batches_list)
        
#     def _calc_padded_batch_length(self, batch_audio_durations):
#         max_audio_duration = max(batch_audio_durations)
#         padded_batch_length = max_audio_duration * len(batch_audio_durations)
#         return padded_batch_length

#     def __iter__(self):
#         return iter(self.batches_list)
        
#     def __len__(self):
#         return len(self.batches_list)