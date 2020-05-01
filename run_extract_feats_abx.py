# Author: David Harwath, Wei-Ning Hsu
import librosa
import numpy as np
import os
import pathlib
import re
import sys
import time
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import DatasetFolder

from dataloaders.utils import compute_spectrogram
from models.AudioModels import ResDavenetVQ
from run_utils import load_audio_model_and_state


def get_dataloader(input_dir, input_ext, batch_size):
    # set target_spec_length to be some large number to avoid truncation
    audio_conf = {'use_raw_length': False, 'target_spec_length': 10000}
    def load_mel_spectrogram_and_path(path):
        y, sr = librosa.load(path, 16000)
        logmelspec, n_frames = compute_spectrogram(y, sr, audio_conf)
        return logmelspec, n_frames, path

    dset = DatasetFolder(input_dir, load_mel_spectrogram_and_path, (input_ext,))
    loader = DataLoader(dset, batch_size=batch_size, shuffle=False, 
                        num_workers=8, pin_memory=True)
    return loader


def write_features_to_text_file(features, path):
    with open(path, 'w') as fp:
        np.savetxt(fp, features.cpu().numpy(), fmt='%.3f')


def extract_and_write_features(audio_model, layer, loader,
                               output_dir, output_name_level):
    vq_layer = re.sub('conv', 'quant', layer)
    get_vq = layer.startswith('quant')
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    def get_output_path(p):
        """
        suppose source audio is `.../a/b/c/d/filename.wav`, if 
        output_name_level=3, the output will be at 
        `<output_dir>/c_d_filename.wav`
        """
        file_base = os.path.splitext(p)[0].split('/')[-output_name_level:]
        file_base = '_'.join(file_base) + '.txt'
        output_path = os.path.join(output_dir, file_base)
        return output_path

    num_utt = 0
    start_time = time.time()
    with torch.no_grad():
        for idx, ((audio_input, n_frames, paths), _) in enumerate(loader):
            audio_input = audio_input.to(device)
            audio_input = audio_input[:, :, :n_frames.max()]

            (_, q_out, preq_out, 
             onehot) = audio_model.get_vq_outputs(audio_input, vq_layer, True)
            audio_output = q_out if get_vq else preq_out
            audio_output = audio_output.squeeze()  # (b, d, n)
            assert(audio_output.dim() == 3)

            ds_ratio = round(audio_input.size(-1) / audio_output.size(-1))
            n_frames.div_(ds_ratio)
            for j, path in enumerate(paths):
                output_path = get_output_path(path)

                # trim paddings and reshape to (n, d)
                utt_n_frames = max(1, n_frames[j])
                utt_audio_output = audio_output[j, :, :utt_n_frames].t()
                write_features_to_text_file(utt_audio_output, output_path)

                num_utt += 1
                if num_utt % 1000 == 0:
                    tot_time = time.time() - start_time
                    print('extracted %d files took %.2fs' % (num_utt, tot_time))

    tot_time = time.time() - start_time
    print('Finished extracting %d files. Took %.2fs' % (num_utt, tot_time))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_path', type=str, help='Load audio model weights from this file.')
    parser.add_argument('input_dir', type=str, help='Get .wav files from this directory')
    parser.add_argument('output_dir', type=str, help='Write .txt files containing embeddings to this directory')
    parser.add_argument('--layer', type=str, default='quant2', help='Which layer to extract features from.')
    parser.add_argument('--input_ext', type=str, default='wav', help='Extension for waveform files to search in input directory')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--output_name_level', type=int, default=1, help='Concat how many levels of original path for output name.')
    args = parser.parse_args()
    print('#' * 15, 'Extract features at %s from %s' % (args.layer, args.model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(8675309)
    
    # Load data and model
    loader =  get_dataloader(args.input_dir, args.input_ext, args.batch_size)
    audio_model = load_audio_model_and_state(state_path=args.model_path)
    audio_model = audio_model.to(device)
    audio_model.eval()

    extract_and_write_features(audio_model, args.layer, loader,
                               args.output_dir, args.output_name_level)
