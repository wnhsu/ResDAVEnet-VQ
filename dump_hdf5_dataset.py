# Author: Wei-Ning Hsu
import h5py
import json
import librosa
import numpy as np
import os
import scipy
import time
from pathlib import Path
from PIL import Image
from torchvision.transforms import transforms

from dataloaders.utils import WINDOWS, compute_spectrogram


def run(json_path, hdf5_json_path, audio_path, image_path, audio_conf={}):
    with open(json_path, 'r') as f:
        data_and_dirs = json.load(f)
        data = data_and_dirs['data']
        audio_base = data_and_dirs['audio_base_path']
        image_base = data_and_dirs['image_base_path']
    print('Loaded %d data from %s' % (len(data), json_path))

    run_audio(data, audio_base, audio_path, audio_conf)
    run_image(data, image_base, image_path)
    
    Path(os.path.dirname(hdf5_json_path)).mkdir(parents=True, exist_ok=True)
    with open(hdf5_json_path, 'w') as f:
        d = {'audio_hdf5_path': audio_path, 'image_hdf5_path': image_path}
        json.dump(d, f)


# Solution borrows from https://github.com/h5py/h5py/issues/745
def run_image(data, image_base, image_path):
    if os.path.exists(image_path):
        print('%s already exists. skip' % image_path)
        return

    print('Dumping image to HDF5 : %s' % image_path)
    n = len(data)
    Path(os.path.dirname(image_path)).mkdir(parents=True, exist_ok=True)
    f = h5py.File(image_path, 'w')
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    dset_img = f.create_dataset('image', (n,), dtype=dt)
    
    start = time.time()
    for i, d in enumerate(data):
        with open('%s/%s' % (image_base, d['image']), 'rb') as f_img:
            binary_img = f_img.read()
        dset_img[i] = np.frombuffer(binary_img, dtype='uint8')

        if i % 100 == 0:
            t = time.time() - start
            print('processed %d / %d images (%.fs)' % (i, n, t))


def run_audio(data, audio_base, audio_path, audio_conf):
    if os.path.exists(audio_path):
        print('%s already exists. skip' % audio_path)
        return

    print('Dumping audio to HDF5 : %s' % audio_path)
    print('  audio_conf : %s' % audio_conf)

    audio_conf['num_mel_bins'] = audio_conf.get('num_mel_bins', 40)
    audio_conf['target_length'] = audio_conf.get('target_length', 2048)
    audio_conf['use_raw_length'] = audio_conf.get('use_raw_length', False)
    assert(not audio_conf['use_raw_length'])
   
    # dump audio
    n = len(data)
    Path(os.path.dirname(audio_path)).mkdir(parents=True, exist_ok=True)
    f = h5py.File(audio_path, 'w')
    dset_mel_shape = (n, audio_conf['num_mel_bins'],
                      audio_conf['target_length'])
    dset_mel = f.create_dataset('melspec', dset_mel_shape, dtype='f')
    dset_len = f.create_dataset('melspec_len', (n,), dtype='i8')

    start = time.time()
    for i, d in enumerate(data):
        y, sr = librosa.load('%s/%s' % (audio_base, d['wav']), None)
        logspec, n_frames = compute_spectrogram(y, sr, audio_conf)
        dset_mel[i, :, :] = logspec
        dset_len[i] = n_frames

        if i % 100 == 0:
            t = time.time() - start
            print('processed %d / %d audios (%.fs)' % (i, n, t))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('inp_json_path', type=str, help='input JSON file')
    parser.add_argument('out_json_path', type=str, help='path to save output json')
    parser.add_argument('audio_h5_path', type=str, help='path to save audio HDF5')
    parser.add_argument('image_h5_path', type=str, help='path to save image HDF5')
    args = parser.parse_args()
    print(args)

    run(args.inp_json_path, args.out_json_path,
        args.audio_h5_path, args.image_h5_path)
