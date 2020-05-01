# Author: David Harwath, Wei-Ning Hsu
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch
import json
import librosa
import numpy as np
import os
import os.path
import scipy.signal
import torch
import torch.nn.functional
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from dataloaders.utils import compute_spectrogram


class ImageCaptionDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf=None, image_conf=None):
        """
        Dataset that manages a set of paired images and audio recordings

        :param dataset_json_file
        :param audio_conf: Dictionary containing the sample rate, window and
        the window length/stride in seconds, and normalization to perform (optional)
        :param image_transform: torchvision transform to apply to the images (optional)
        """
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.data = data_json['data']

        self.image_base_path = data_json.get('image_base_path', '')
        self.audio_base_path = data_json.get('audio_base_path', '')

        self.audio_conf = audio_conf if audio_conf else {}
        self.image_conf = image_conf if image_conf else {}

        # image transforms
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

    def _LoadAudio(self, path):
        y, sr = librosa.load(path, None)
        logspec, n_frames = compute_spectrogram(y, sr, self.audio_conf)
        return logspec, n_frames

    def _LoadImage(self, impath):
        img = Image.open(impath).convert('RGB')
        img = self.image_resize_and_crop(img)
        img = self.image_normalize(img)
        return img

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        datum = self.data[index]
        audio, nframes = self._LoadAudio(os.path.join(self.audio_base_path, datum['wav']))
        image = self._LoadImage(os.path.join(self.image_base_path, datum['image']))
        return image, audio, nframes

    def __len__(self):
        return len(self.data)
