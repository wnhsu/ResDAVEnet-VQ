# Authors: Wei-Ning Hsu, David Harwath

import argparse
import numpy as np
import os
import pickle
import sys
import time
import torch
from collections import defaultdict, Counter
from itertools import groupby

import dataloaders
import models
from .alignment import SparseAlignment, DenseAlignment, align_sparse_to_dense


STOP_WORDS = [
    "A", "AN", "THE", "AND", "OF", "<SPOKEN_NOISE>",
    "TO", "ON", "IN", "AT", "ARE", "IS", 
    "IT", "HE", "SHE", "YOU", "THERE", "THIS"
]


###############
# Model Utils #
###############
def get_embedding(audio_model, layer):
    """
    Return:
        embedding (torch.Tensor) : of shape (num_codes, embedding_dim)
    """
    embedding = audio_model._modules[layer]._embedding.weight.data.cpu()
    return embedding


def onehots2codes(onehots):
    return torch.stack([torch.argmax(x) for x in onehots])


def get_feats_codes(audio_model, layer, mels, device, mel_spf=0.01):
    """
    Return:
        feats (torch.Tensor) : (num_frames, feat_dim)
        pre_feats (torch.Tensor) : (num_frames, feat_dim)
        codes (torch.Tensor) : (num_frames,)
        spf (float) : second per frame for features and codes
    """
    with torch.no_grad():
        mels = mels.unsqueeze(0).to(device)

        _, feats, pre_feats, onehots = audio_model.get_vq_outputs(
                mels, layer, unflatten=True)
        spf = mel_spf * round(mels.size(-1) / feats.size(-1))
        feats = feats.squeeze().permute(1, 0)
        codes = onehots2codes(onehots.squeeze().permute(1, 0))
        pre_feats = pre_feats.squeeze().permute(1, 0)
    return feats, pre_feats, codes, spf


def comp_code_to_wordprec(utt2codes, utt2words, code_halfwidth, stop_words):
    """
    Return
        code_to_wordprec (dict) : code_to_wordprec[code] is a list of (word,
            precision, num_occ) sorted by precision
    """
    def center_to_range(center):
        s = center - code_halfwidth
        e = center + code_halfwidth
        return s, e

    ts = time.time()
    code_to_nsegs = defaultdict(int)
    code_to_wordcounts = defaultdict(Counter)
    for utt_index in utt2codes:
        code_ali = utt2codes[utt_index]    
        word_ali = utt2words[utt_index]
        if word_ali is None:
            continue
        for code, seg_wordset in align_sparse_to_dense(word_ali, code_ali,
                                                       center_to_range):
            code_to_nsegs[code] += 1
            code_to_wordcounts[code].update(seg_wordset)
    # print('get code_to_wordcounts takes %.fs' % (time.time()-ts))

    code_to_wordprec = dict()     
    n_codes_with_no_words = 0
    for code, nsegs in sorted(code_to_nsegs.items()):
        word_counts = code_to_wordcounts[code]
        word_prec_list = [(word, float(occ)/nsegs, occ) for word, occ \
                          in word_counts.most_common() if word not in stop_words]
        n_codes_with_no_words += int(not bool(word_prec_list))
        code_to_wordprec[code] = word_prec_list
    
    print('%d / %d codes mapped to only utterances with empty transcripts' % (
          n_codes_with_no_words, len(code_to_nsegs)))
    return code_to_wordprec


def get_high_prec_words(code_to_wordprec, threshold=0.35):
    word_to_maxprec = defaultdict(float)
    for word2prec in code_to_wordprec.values():
        for word, prec, _ in word2prec:
            word_to_maxprec[word] = max(prec, word_to_maxprec[word])
    
    high_prec_words = {w for w, p in word_to_maxprec.items() if p >= threshold}
    print("#"*5, "Total %d words with precision >= %s"
          % (len(high_prec_words), threshold))
    return high_prec_words


def comp_word_to_coderecall(utt2codes, utt2words, target_words):
    """
    Compute recall of given words. If `target_words == []`, compute all words.
    
    Return
        word_to_coderecall (dict) : word_to_coderecall[word] is a list of
            (code, recall, num_occ) sorted by recall
    """
    ts = time.time()
    word_to_nsegs = defaultdict(int)
    word_to_codecounts = defaultdict(Counter)    
    for utt_index in utt2codes:
        code_ali = utt2codes[utt_index]
        word_ali = utt2words[utt_index]
        if word_ali is None:
            continue
        for word_s, word_e, word in word_ali.data:
            if target_words and (word not in target_words):
                continue
            seg_codeset = set(code_ali.get_segment(word_s, word_e))
            word_to_nsegs[word] += 1
            word_to_codecounts[word].update(seg_codeset)
    # print('get word_to_codecounts takes %.fs' % (time.time()-ts))

    word_to_coderecall = dict()
    for word, nsegs in word_to_nsegs.items():
        code_counts = word_to_codecounts[word]
        code_recall_list = [(code, float(occ)/nsegs, occ) for code, occ \
                            in code_counts.most_common()]
        word_to_coderecall[word] = code_recall_list

    return word_to_coderecall


def comp_code_norms(audio_model, layer):
    return get_embedding(audio_model, layer).norm(dim=1, p=2).numpy()


def comp_code_word_f1(code_to_wordprec, word_to_coderecall, min_occ):
    """
    Returns:
        code_to_wordf1 (dict) : code maps to a list of (word, f1, prec, recall, occ)
        word_to_codef1 (dict) : word maps to a list of (code, f1, prec, recall, occ)
    """
    code_to_word2prec = {}
    for code in code_to_wordprec:
        wordprec = code_to_wordprec[code]
        code_to_word2prec[code] = {word : prec for word, prec, _ in wordprec}

    word_to_code2recall = {}
    for word in word_to_coderecall:
        coderecall = word_to_coderecall[word]
        word_to_code2recall[word] = {code : (recall, occ) \
                                     for code, recall, occ in coderecall}

    code_to_wordf1 = defaultdict(list)
    for code in code_to_word2prec:
        for word, prec in code_to_word2prec[code].items():
            recall, occ = word_to_code2recall.get(word, {}).get(code, (0, 0))
            if occ >= min_occ:
                f1 = compute_f1(prec, recall)
                code_to_wordf1[code].append((word, f1, prec, recall, occ))
        code_to_wordf1[code] = sorted(code_to_wordf1[code], key=lambda x: -x[1])

    word_to_codef1 = defaultdict(list)
    for word in word_to_code2recall:
        for code, (recall, occ) in word_to_code2recall[word].items():
            if occ >= min_occ:
                prec = code_to_word2prec.get(code, {}).get(word, 0)
                f1 = compute_f1(prec, recall)
                word_to_codef1[word].append((code, f1, prec, recall, occ))
        word_to_codef1[word] = sorted(word_to_codef1[word], key=lambda x: -x[1])

    return code_to_wordf1, word_to_codef1


###########
# Metrics #
###########
def compute_f1(prec, recall):
    return 2*prec*recall / (prec+recall)


###############
# Print Utils #
###############
def print_code_to_word_prec(code_to_wordprec, prec_threshold=0.35,
                            num_show=3, show_all=True):
    n_codes = len(code_to_wordprec.keys())
    n_codes_above_prec_threshold = 0
            
    for code in sorted(code_to_wordprec.keys()):
        wordprec = code_to_wordprec[code]
        if not len(wordprec):
            continue
        (top_word, top_prec, _) = wordprec[0]
        above_prec_threshold = (top_prec >= prec_threshold)
        if above_prec_threshold:
            n_codes_above_prec_threshold += 1
            
        if show_all or above_prec_threshold:
            tot_occ = sum([occ for _, _, occ in wordprec])
            # show top-k
            msg = "%s %4d (#words=%5d, occ=%5d): " % (
                "*" if above_prec_threshold else " ", 
                code, len(wordprec), tot_occ)
            for word, prec, _ in wordprec[:num_show]:
                res = "%s (%5.2f)" % (word, prec)
                msg += " %-25s|" % res
            print(msg)
    
    print(('Found %d / %d (%.2f%%) codes with a word detector with' 
           'prec greater than %f.') % (
            n_codes_above_prec_threshold, n_codes, 
            n_codes_above_prec_threshold / n_codes * 100, prec_threshold)) 


def print_word_by_code_recall(word_to_coderecall, num_show=3):
    for word in sorted(word_to_coderecall):
        tot_occ = sum([o for _, _, o in word_to_coderecall[word]])
        print("%-15s (#occ = %4d)" % (word, tot_occ),
              [('%4d' % c, '%.2f' % r) for c, r, _ in word_to_coderecall[word][:num_show]])


def print_code_stats_by_f1(code_to_wordf1, code_ranks_show=range(10),
                           num_word_show=2):
    print("##### Showing ranks %s" % str(code_ranks_show))
    codes = sorted(code_to_wordf1.keys(), 
                   key=lambda x: (-code_to_wordf1[x][0][1] if len(code_to_wordf1[x]) else 0))
    print('%3s & %4s & %10s & %6s & %6s & %6s & %5s'
          % ('rk', 'code', 'word', 'F1', 'Prec', 'Recall', 'Occ'))
    for rank in code_ranks_show:
        if rank >= len(codes):
            continue
        code = codes[rank]
        msg = '%3d & %4d' % (rank+1, code)
        for word, f1, prec, recall, occ in code_to_wordf1[code][:num_word_show]:
            msg += ' & %10s & %6.2f & %6.2f & %6.2f & %5d' % (
                    word.lower(), f1*100, prec*100, recall*100, occ)
        msg += ' \\\\'
        print(msg)


def print_word_stats_by_f1(word_to_codef1, word_ranks_show=range(10),
                           num_code_show=3):
    print("##### Showing ranks %s" % str(word_ranks_show))
    words = sorted(word_to_codef1.keys(), 
                   key=lambda x: (-word_to_codef1[x][0][1] if len(word_to_codef1[x]) else 0))
    print('%3s & %15s & %4s & %6s & %6s & %6s & %5s'
          % ('rk', 'word', 'code', 'F1', 'Prec', 'Recall', 'Occ'))
    for rank in word_ranks_show:
        if rank >= len(words):
            continue
        word = words[rank]
        msg = '%3d & %15s' % (rank+1, word.lower())
        for code, f1, prec, recall, occ in word_to_codef1[word][:num_code_show]:
            msg += ' & %4d & %6.2f & %6.2f & %6.2f & %5d' % (
                    code, f1*100, prec*100, recall*100, occ)
        msg += ' \\\\'
        print(msg)
    

def count_high_f1_words(word_to_codef1, f1_threshold=0.5, verbose=True):
    count = 0 
    for word in word_to_codef1.keys():
        if len(word_to_codef1[word]) and (word_to_codef1[word][0][1] >= f1_threshold):
            count += 1
    if verbose:
        print('%d / %d words with an F1 score >= %s'
              % (count, len(word_to_codef1), f1_threshold))
    return count
    

def compute_topk_avg_f1(word_to_codef1, k=250, verbose=True):
    f1s = [word_to_codef1[word][0][1] for word in word_to_codef1 \
           if len(word_to_codef1[word])]
    top_f1s = sorted(f1s, reverse=True)[:k]
    
    if verbose:
        print('avg F1 = %.2f%% for top %d words; %.2f%% for all %d words'
              % (100*np.mean(top_f1s), len(top_f1s), 100*np.mean(f1s), len(f1s)))
    return 100*np.mean(top_f1s)


def print_analysis(code_to_wordprec, word_to_coderecall,
                   code_norms, high_prec_words, min_occ, rank_range):
    print_code_to_word_prec(code_to_wordprec, prec_threshold=0.35,
                            num_show=3, show_all=True)
    print_word_by_code_recall(word_to_coderecall, num_show=3)

    code_to_wordf1, word_to_codef1 = comp_code_word_f1(
            code_to_wordprec, word_to_coderecall, min_occ=min_occ)
    print_code_stats_by_f1(code_to_wordf1, rank_range)
    print_word_stats_by_f1(word_to_codef1, rank_range)
    count_high_f1_words(word_to_codef1, f1_threshold=0.5)
    compute_topk_avg_f1(word_to_codef1, k=250)
