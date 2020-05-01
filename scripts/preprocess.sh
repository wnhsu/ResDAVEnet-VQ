#!/bin/bash 

# Author: Wei-Ning Hsu


python dump_hdf5_dataset.py \
  "./filelist/train_original_with_alignments.json" \
  "./data/PlacesEnglish400kTrainHDF5.json" \
  "./data/PlacesEnglish400kTrain_audio.hdf5" \
  "./data/PlacesEnglish400kTrain_image.hdf5"

python dump_hdf5_dataset.py \
  "./filelist/val_original_with_alignments.json" \
  "./data/PlacesEnglish400kValHDF5.json" \
  "./data/PlacesEnglish400kVal_audio.hdf5" \
  "./data/PlacesEnglish400kVal_image.hdf5"
