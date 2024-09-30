import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import logfbank, mfcc

def get_audio_repersentetions(base_dir, df, classes):
    signals = {}
    fft = {}
    fbank = {}
    mfccs = {}
    trained_mfccs = {}
    specs = {}
    stfts = {}
    for c in list(classes):
        wav_file = df[df.label == c].index[0]
        rate, signal = scipy.io.wavfile.read(os.path.join(base_dir,'audio_train', wav_file))
        mask = envelope(signal, rate, 0.0005)
        signal = signal[mask]
        signals[c] = signal
        fft[c] = calc_fft(signal, rate)
        bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
        fbank[c] = bank
        mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
        mfccs[c] = mel
        s = calc_spec(signal, rate)
        specs[c] = s[:]
    return signals, mfccs, specs

def plot_labels_dist(label_counts):
    fig, ax = plt.subplots()
    ax.set_title("class dist", y=1.08)
    ax.pie(
        label_counts.values(), labels=label_counts.keys(), autopct="%1.1f%%", shadow=False, startangle=90
    )
    ax.axis("equal")
    plt.show()
    
def plot_signals(signals):
    num_rows = int(np.ceil(len(signals) / 5))
    fig, axes = plt.subplots(nrows=num_rows, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(num_rows):
        for y in range(5):
            if num_rows > 1:
                cur_ax = axes[x,y]
            else: 
                cur_ax = axes[y]
            cur_ax.set_title(list(signals.keys())[i])
            cur_ax.plot(list(signals.values())[i])
            cur_ax.get_xaxis().set_visible(False)
            cur_ax.get_yaxis().set_visible(False)
            i += 1

def plot_stfts(stfts):
    num_rows = int(np.ceil(len(stfts) / 5))
    fig, axes = plt.subplots(nrows=num_rows, ncols=5, sharex=False,
                             sharey=True, figsize=(80,20))
    fig.suptitle('Short-Time fourier transform', size=40)
    i = 0
    for x in range(num_rows):
        for y in range(5):
            if num_rows > 1:
                cur_ax = axes[x,y]
            else: 
                cur_ax = axes[y]
            cur_ax.set_title(list(stfts.keys())[i])
            cur_stft = list(stfts.values())[i]
            cur_ax.imshow(cur_stft.get_array(),
                    cmap='hot', interpolation='nearest')
            cur_ax.get_xaxis().set_visible(False)
            cur_ax.get_yaxis().set_visible(False)
            i += 1

            
def get_length(a_id):
    rate, signal = scipy.io.wavfile.read("../wavfiles/" + a_id)
    return signal.shape[0] / rate

def envelope(y: pd.Series, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate / 10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1 / rate)
    Y = abs(np.fft.rfft(y) / n)
    return Y, freq


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]
        
        
def calc_spec(y, rate):
    spec = []
    for y in chunks(y, int(len(y) / 100)):
        fft_res = calc_fft(y, rate)[0]
        if len(spec) > 2:
            if len(fft_res) == len(spec[-1]):
                spec.append(list(fft_res))
        else:
            spec.append(list(fft_res))

    return np.array(spec)


def plot_principles_components(S):
    fig, ax = plt.subplots()
    principle_compenets = np.arange(len(S)) + 1
    ax.bar(principle_compenets, S.detach().numpy(), tick_label=principle_compenets)
    ax.set_xlabel("Component index")
    ax.set_ylabel("Component Energy")
    ax.set_title("Energy of principle compenets")
    plt.show()
    
    
def scatter_embeddings_clusters(projected_embeddings, targets, classes):
    for i in range(len(classes)):
        target_indexes = (targets == i)
        if sum(target_indexes) > 0:
            first_componenet_projection = (
                projected_embeddings[target_indexes, 0].detach().numpy()
            )
            second_componenet_projection = (
                projected_embeddings[target_indexes, 1].detach().numpy()
            )
            plt.scatter(
                first_componenet_projection,
                second_componenet_projection,
                label=classes[i],
            )
    plt.legend()
    
    
def random_sample_data(embeddings, targets, sample_data_size):
    indexes = np.random.choice(np.arange(targets.shape[0]), sample_data_size, replace=False)
    sampled_embeddings = embeddings.detach().numpy()[indexes, :]
    sample_targets = targets[indexes]
    return sampled_embeddings, sample_targets


def plot_TSNE(embeddings, targets, classes):
    reducer = TSNE(n_components=2)
    reducer_result = reducer.fit_transform(embeddings)
    scatter = plt.scatter(
        reducer_result[:, 0], reducer_result[:, 1], c=targets, cmap="jet")
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plt.show()