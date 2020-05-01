# Author: Wei-Ning Hsu
import librosa
import numpy as np
import scipy.signal
import torch


WINDOWS = {'hamming': scipy.signal.hamming,
           'hann': scipy.signal.hann,
           'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}

def compute_spectrogram(y, native_sample_rate, audio_conf={}):
    # Default audio configuration
    audio_type = audio_conf.get('audio_type', 'melspectrogram')
    if audio_type not in ['melspectrogram', 'spectrogram']:
        raise ValueError('Invalid audio_type specified in audio_conf.')

    preemph_coef = audio_conf.get('preemph_coef', 0.97)
    sample_rate = audio_conf.get('sample_rate', 16000)
    window_size = audio_conf.get('window_size', 0.025)
    window_stride = audio_conf.get('window_stride', 0.01)
    window_type = audio_conf.get('window_type', 'hamming')
    n_fft = audio_conf.get('n_fft', int(sample_rate * window_size))

    num_mel_bins = audio_conf.get('num_mel_bins', 40)
    fmin = audio_conf.get('fmin', 20)

    target_length = audio_conf.get('target_length', 2048)
    use_raw_length = audio_conf.get('use_raw_length', False)
    padval = audio_conf.get('padval', 0)
    
    win_length = int(sample_rate * window_size)
    hop_length = int(sample_rate * window_stride)

    # resample to the target sample rate
    y = librosa.resample(y, native_sample_rate, sample_rate)

    # subtract DC, preemphasis
    if y.size == 0:
        y = np.zeros(200)
    y = y - y.mean()
    y = np.append(y[0],y[1:]-preemph_coef*y[:-1])
    
    # compute spectrogram
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                        win_length=win_length,
                        window=WINDOWS[window_type])
    spec = np.abs(stft)**2
    if audio_type == 'melspectrogram':
        mel_basis = librosa.filters.mel(sample_rate, n_fft, n_mels=num_mel_bins,
                                        fmin=fmin)
        spec = np.dot(mel_basis, spec)
    logspec = librosa.power_to_db(spec, ref=np.max)

    # optional trimming/padding
    n_frames = logspec.shape[1]
    if use_raw_length:
        target_length = n_frames
    p = target_length - n_frames
    if p > 0:
        logspec = np.pad(logspec, ((0,0),(0,p)), 'constant',
                         constant_values=(padval,padval))
    elif p < 0:
        print('WARNING: truncate %d/%d frames' % (-p, n_frames))
        logspec = logspec[:,0:p]
        n_frames = target_length

    logspec = torch.FloatTensor(logspec)
    return logspec, n_frames


