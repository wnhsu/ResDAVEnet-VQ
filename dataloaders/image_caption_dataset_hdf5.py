# Author: Wei-Ning Hsu
import h5py
import io
import json
import numpy as np
import os
import torch
import torch.nn.functional
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class ImageCaptionDatasetHDF5(Dataset):
    def __init__(self, json_path, audio_conf={}, image_conf={}):
        """
        Dataset that manages a set of paired images and audio recordings

        :param audio_hdf5 (str): path to audio hdf5
        :param image_hdf5 (str): path to audio hdf5
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.audio_hdf5_path = data['audio_hdf5_path']
        self.image_hdf5_path = data['image_hdf5_path']
        
        # delay creation until __getitem__ is first called
        self.audios = None
        self.images = None
        
        # audio features are pre-computed with default values.
        self.audio_conf = audio_conf
        self.image_conf = image_conf

        # load image/audio dataset size and check if number matches
        audios = h5py.File(self.audio_hdf5_path, 'r')
        images = h5py.File(self.image_hdf5_path, 'r')
        assert audios['melspec'].shape[0] == images['image'].shape[0]
        self.n_samples = audios['melspec'].shape[0]
        audios.close()
        images.close()

        crop_size = self.image_conf.get('crop_size', 224)
        center_crop = self.image_conf.get('center_crop', False)
        if center_crop:
            self.image_resize_and_crop = transforms.Compose(
                [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        else:
            self.image_resize_and_crop = transforms.Compose(
                [transforms.RandomResizedCrop(crop_size), transforms.ToTensor()])

        RGB_mean = self.image_conf.get('RGB_mean', [0.485, 0.456, 0.406])
        RGB_std = self.image_conf.get('RGB_std', [0.229, 0.224, 0.225])
        self.image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

    def _LoadAudio(self, index):
        if self.audios is None:
            print('Loading audio from %s' % self.audio_hdf5_path)
            self.audios = h5py.File(self.audio_hdf5_path, 'r')
        n_frames = self.audios['melspec_len'][index]
        logspec = self.audios['melspec'][index]
        logspec = torch.FloatTensor(logspec)
        if self.audio_conf.get('normalize', False):
            mean = logspec[:, 0:n_frames].mean()
            std = logspec[:, 0:n_frames].std()
            logspec[:, 0:n_frames].add_(-mean)
            logspec[:, 0:n_frames].div_(std)
        return logspec, n_frames

    def _LoadImage(self, index):
        if self.images is None:
            print('Loading image from %s' % self.image_hdf5_path)
            self.images = h5py.File(self.image_hdf5_path, 'r')
        binary_img = self.images['image'][index]
        img = Image.open(io.BytesIO(binary_img)).convert('RGB')
        img = self.image_resize_and_crop(img)
        img = self.image_normalize(img)
        return img

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) 
        nframes is an integer
        """
        audio, nframes = self._LoadAudio(index)
        image = self._LoadImage(index)
        return image, audio, nframes

    def __len__(self):
        return self.n_samples
