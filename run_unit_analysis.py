# Author: Wei-Ning Hsu, David Harwath
import argparse
import json
import numpy as np
import os
import pickle
import sys
import time
import torch

import dataloaders
import models
from steps.unit_analysis import (comp_code_to_wordprec, comp_word_to_coderecall,
                                 get_high_prec_words, comp_code_norms,
                                 print_analysis, get_feats_codes,
                                 SparseAlignment, DenseAlignment, STOP_WORDS)
from run_utils import load_audio_model_and_state


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################
# Dataset Utils #
#################
def get_inp_feat(dataset, index):
    mels, nframes = dataset._LoadAudio(index)
    mels = mels[:, :nframes]
    # print('FEAT : idx=%d, mel=%s, nframes=%s' % (index, mels.size(), nframes))
    return mels, nframes


def get_word_ali(json_data, index):
    """
    raw_ali is a string like 'start1__word1__end1 start2__word2__end2 ...'
    """
    raw_ali = json_data[index].get('text_alignment', None)
    if raw_ali is None:
        return None
    
    data = []
    meta_toks = raw_ali.split()
    for meta_tok in meta_toks:
        toks = meta_tok.split('__')
        if len(toks) == 3:
            data.append((float(toks[0]), float(toks[2]), toks[1]))
    
    if len(data) == 0:
        return None
    else:
        return SparseAlignment(data)


def get_code_ali(audio_model, layer, dataset, index, device):
    mels, _ = get_inp_feat(dataset, index)
    _, _, codes, spf = get_feats_codes(audio_model, layer, mels, device)
    code_list = codes.detach().cpu().tolist()
    return DenseAlignment(code_list, spf)


###########################
# Prepare data and outpus #
###########################
def prepare_data(audio_model, layer, dataset, json_data, max_n_utts):
    utt2codes = {}
    utt2words = {}
    tot_nframes = 0
    n_utts = min(len(dataset), max_n_utts)
    
    t0 = time.time()
    for utt_index in range(n_utts):
        utt2codes[utt_index] = get_code_ali(audio_model, layer, dataset,
                                            utt_index, device)
        utt2words[utt_index] = get_word_ali(json_data, utt_index)
        tot_nframes += len(utt2codes[utt_index])
        if (utt_index+1) % 1000 == 0:
            print("processing %d / %d; accumulated %d frames; took %.fs" 
                  % (utt_index+1, len(dataset), tot_nframes, time.time()-t0),
                  flush=True)
    t1 = time.time()
    
    print('')
    print('Took %.fs to dump %d code frames for %d utts'
          % (t1 - t0, tot_nframes, n_utts))
    return utt2codes, utt2words


#############
# Exp Utils #
#############
def load_dataset(json_path, hdf5_path):
    dataset = dataloaders.ImageCaptionDatasetHDF5(hdf5_path, audio_conf={})
    with open(json_path) as f:
        json_data = json.load(f)['data']
    return dataset, json_data


def run_analysis(utt2codes, utt2words, stop_words, output_dir):
    code_to_wordprec = load_or_run_and_dump(
            '%s/code_to_wordprec.pkl' % output_dir,
            comp_code_to_wordprec,
            [utt2codes, utt2words, 0.04, stop_words], {})

    word_to_coderecall = load_or_run_and_dump(
            '%s/word_to_coderecall.pkl' % output_dir,
            comp_word_to_coderecall,
            [utt2codes, utt2words, []], {})

    high_prec_words = load_or_run_and_dump(
            '%s/high_prec_words.pkl' % output_dir,
            get_high_prec_words,
            [code_to_wordprec], {'threshold': 0.35})
    return code_to_wordprec, word_to_coderecall, high_prec_words


def load_or_run_and_dump(filename, fn, args, kwargs):
    if os.path.exists(filename):
        print('Load from %s' % filename)
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        print('Run %s and dump to %s' % (fn.__name__, filename))
        x = fn(*args, **kwargs)
        with open(filename, 'wb') as f:
            pickle.dump(x, f)
        return x


if __name__ == '__main__':
    print('Current time : %s' % time.asctime())
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', default='', type=str, required=True)
    parser.add_argument('--json_path', default='', type=str, required=True)
    parser.add_argument('--hdf5_path', default='', type=str, required=True)
    parser.add_argument('--layer', default='quant3', type=str)
    parser.add_argument('--max_n_utts', default=20000, type=int)
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--min_occ', default=1, type=int)
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    max_n_utts = args.max_n_utts
    min_occ = args.min_occ
    layer = args.layer
    
    # Load data and model
    dataset, json_data = load_dataset(args.json_path, args.hdf5_path)
    audio_model = load_audio_model_and_state(exp_dir=args.exp_dir)
    audio_model = audio_model.to(device)
    audio_model.eval()
    print(audio_model)

    # run
    utt2codes, utt2words = load_or_run_and_dump(
            '%s/utt2codes_utt2words.pkl' % output_dir,
            prepare_data, [audio_model, layer, dataset, json_data, max_n_utts], {})
    code_norms = load_or_run_and_dump(
            '%s/code_norms.pkl' % output_dir,
            comp_code_norms, [audio_model, layer], {})

    c2wp, w2cr, hp_ws = run_analysis(utt2codes, utt2words,
                                     STOP_WORDS, output_dir)
    
    rank_range = list(range(40)) + list(range(180, 200))
    print_analysis(c2wp, w2cr, code_norms, hp_ws, min_occ, rank_range)
