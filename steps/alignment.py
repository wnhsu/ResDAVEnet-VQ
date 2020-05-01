# Author: Wei-Ning Hsu

import numpy as np
import torch


class Alignment(object):
    def __init__(self):
        raise

    def __len__(self):
        return len(self.data)

    @property
    def data(self):
        return self._data


class SparseAlignment(Alignment):
    """
    alignment is a list of (start_time, end_time, value) tuples.
    """
    def __init__(self, data, unit=1.):
        self._data = [(s*unit, e*unit, v) for s, e, v in data]

    def __repr__(self):
        return str(self._data)

    def get_segment(self, seg_s, seg_e, empty_word='<SIL>'):
        """
        return words in the given segment.
        """
        seg_ali = self.get_segment_ali(seg_s, seg_e, empty_word)
        return [word for _, _, word in seg_ali.data]

    def get_segment_ali(self, seg_s, seg_e, empty_word=None, contained=False):
        seg_data = []
        if contained:
            is_valid = lambda s, e: (s >= seg_s and e <= seg_e)
        else:
            is_valid = lambda s, e: (max(s, seg_s) < min(e, seg_e))

        for (word_s, word_e, word) in self.data:
            if is_valid(word_s, word_e):
                seg_data.append((word_s, word_e, word))
        if not seg_data and empty_word is not None:
            seg_data = [(seg_s, seg_e, empty_word)]
        return SparseAlignment(seg_data)

    def get_words(self):
        return {word for _, _, word in self.data}

    def has_words(self, check_words):
        """check_words is assumed to be a set"""
        assert(isinstance(check_words, set))
        return bool(self.get_words().intersection(check_words))


class DenseAlignment(Alignment):
    """
    alignment is a list of values that is assumed to have equal duration.
    """
    def __init__(self, data, spf, offset=0):
        assert(offset >= 0)
        self._data = data
        self._spf = spf
        self._offset = offset
    
    @property
    def spf(self):
        return self._spf

    @property
    def offset(self):
        return self._offset

    def __repr__(self):
        return 'offset=%s, second-per-frame=%s, data=%s' % (self._offset, self._spf, self._data)

    def get_center(self, frm_index):
        return (frm_index + 0.5) * self.spf + self.offset

    def get_segment(self, seg_s, seg_e):
        """
        return words in the given segment
        """
        seg_frm_s = (seg_s - self.offset) / self.spf
        seg_frm_s = int(max(np.floor(seg_frm_s), 0))

        seg_frm_e = (seg_e - self.offset) / self.spf
        seg_frm_e = int(min(np.ceil(seg_frm_e), len(self.data)))
        
        seg_words = self.data[seg_frm_s:seg_frm_e]
        return seg_words

    def get_ali_and_center(self):
        """return a list of (code, center_time_sec)"""
        return [(v, self.get_center(f)) \
                for f, v in enumerate(self.data)]

    def get_sparse_ali(self):
        new_data = list(self.data) + [-1]
        changepoints = [j for j in range(1, len(new_data)) \
                        if new_data[j] != new_data[j-1]]
    
        prev_cp = 0
        sparse_data = []
        for cp in changepoints:
            t_s = prev_cp * self._spf + self._offset
            t_e = cp * self._spf + self._offset
            sparse_data.append((t_s, t_e, new_data[prev_cp]))
            prev_cp = cp
        return SparseAlignment(sparse_data)


##############################
# Transcript Post-Processing #
##############################

def align_sparse_to_dense(sp_ali, dn_ali, center_to_range):
    """
    ARGS:
        sp_ali (SparseAlignment):
        dn_ali (DenseAlignment):
    """
    ret = []
    w_s_list, w_e_list, w_list = zip(*sp_ali.data)
    w_sidx = 0  # first word that the current segment's start is before a word's end
    w_eidx = 0  # first word that the current segment's end is before a word's start
    for code, cs in dn_ali.get_ali_and_center():
        ss, es = center_to_range(cs)
        while w_sidx < len(w_list) and ss > w_e_list[w_sidx]:
            w_sidx += 1
        while w_eidx < len(w_list) and es > w_s_list[w_eidx]:
            w_eidx += 1
        seg_wordset = set(w_list[w_sidx:w_eidx]) if w_eidx > w_sidx else {'<SIL>'}
        ret.append((code, seg_wordset))
    return ret

