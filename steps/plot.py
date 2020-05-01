# Authors: Wei-Ning Hsu, David Harwath

import librosa
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import scipy.signal
import seaborn as sns
from matplotlib.lines import Line2D

from dataloaders.utils import compute_spectrogram


sns.set_style('darkgrid')


def load_raw_spectrogram(path):
    audio_conf = {'audio_type': 'spectrogram',
                  'sample_rate': 16000,
                  'window_size': 0.005,
                  'window_stride': 0.002,
                  'n_fft': 1024,
                  'use_raw_length': True}
    y, sr = librosa.load(path, audio_conf['sample_rate'])
    logspec, _ = compute_spectrogram(y, sr, audio_conf)
    logspec = logspec.numpy()
    return logspec, audio_conf['window_stride'], audio_conf['sample_rate']


def plot_one_ali(ax, ali, fontsize=12, ybot_r=0, ytop_r=0.1, color='red', valter=False, xleft_r=0):
    (ymin, ymax) = ax.get_ylim()
    (xmin, xmax) = ax.get_xlim()
    ybot = (1 + ybot_r) * ymax
    ytop = (1 + ytop_r) * ymax
    
    # horizontal lines
    botline = Line2D([xmin, xmax], [ybot, ybot], color='black')
    topline = Line2D([xmin, xmax], [ytop, ytop], color='black')    
    botline.set_clip_on(False)
    topline.set_clip_on(False)
    ax.add_line(botline)
    ax.add_line(topline)
    
    # text location parameters
    sign = 1
    ydelta = (ytop - ybot)*0.2
    ymid = (ybot*0.6 + ytop*0.4)
    
    for (t_s, t_e, lab) in ali:
        # add label
        text_xs = t_s + (t_e - t_s) * 0.1
        text_ys = ymid + sign*ydelta if valter else ymid
        sign = sign*-1
        ax.text(text_xs, text_ys, lab, fontsize=fontsize)
        
        # add label boundaries
        vline_s = Line2D([t_s, t_s], [ybot, ytop], ls='dashed', color=color, alpha=0.3)
        vline_e = Line2D([t_e, t_e], [ybot, ytop], ls='dashed', color=color, alpha=0.3)
        vline_s.set_clip_on(False)
        vline_e.set_clip_on(False)        
        ax.add_line(vline_s)
        ax.add_line(vline_e)


def plot_spec_and_alis(wav_path, alis, seg_s=0, seg_e=-1, fontsize=12, ali_height=0.2, 
                       valter=True, colors=['b', 'r', 'g', 'y'], figsize=(16,8)):
    spec, spf, sr = load_raw_spectrogram(wav_path)
    nsec = spf*spec.shape[1]
    
    seg_e = nsec if seg_e == -1 else min(nsec, seg_e)
    if seg_s != 0 or seg_e != nsec:
        spec = spec[:, int(seg_s/spf):int(seg_e/spf)]
        alis = [ali.get_segment_ali(seg_s, seg_e, contained=True) for ali in alis]

    # visualize alignment
    plt.figure(figsize=figsize)
    plt.imshow(np.flipud(-1 * spec), aspect='auto', cmap='gray',
               extent=[seg_s, seg_e, 0, sr/2])

    ax = plt.gca()
    for idx, ali in enumerate(alis):
        color = colors[idx % len(colors)]
        plot_one_ali(ax, ali.data, fontsize=fontsize, 
                     ybot_r=idx*ali_height, ytop_r=(idx+1)*ali_height,
                     color=color, valter=valter)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')

    plt.tight_layout()
    plt.show()


def plot_precision_recall(word_to_codef1, num_words=250, ax=None):
    def get_f1(word):
        if len(word_to_codef1[word]):
            return word_to_codef1[word][0][1]
        return 0
    
    words = sorted(word_to_codef1.keys(), key=get_f1, reverse=True)
    words = words[:num_words]
    
    f1s = [word_to_codef1[word][0][1] for word in words]
    precs = [word_to_codef1[word][0][2] for word in words]
    recalls = [word_to_codef1[word][0][3] for word in words]

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(precs, recalls, alpha=0.5)
    ax.set_xlabel("Precision", fontsize=14)
    ax.set_ylabel("Recall", fontsize=14)
    ax.set_title('Words of Top %d F1-Scores' % num_words, fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    

def plot_num_words_above_f1_threshold(word_to_codef1, f1_thresholds, ax=None):
    f1s = [word_to_codef1[x][0][1] for x in word_to_codef1 if word_to_codef1[x]]
    num_words_above_f1_thresholds = [sum(f1s >= x) for x in f1_thresholds]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(f1_thresholds, num_words_above_f1_thresholds)
    ax.set_xlabel("F1 Threshold", fontsize=14)
    ax.set_ylabel("Number of Words", fontsize=14)
    ax.set_title('Number of Words above F1 Threshold', fontsize=14)
    ax.set_yscale('log')
