#!/bin/bash 

# Author: Wei-Ning Hsu


vqon=$1    # 01000
name=$2    # RDVQ_00000_01000
args=$3    # "--resume True --seed_dir <dir>"

data_tr="./data/PlacesEnglish400kTrainHDF5.json"
data_dt="./data/PlacesEnglish400kValHDF5.json"
vqonarg=$(echo $vqon | sed 's/./&,/g' | sed 's/,$//g')  # insert ',' in between

expdir="./exps/$name"
echo "DIRECTORY: $expdir"
[ -d "$expdir" ] || mkdir -p "$expdir"

python run_ResDavenetVQ.py --mode train \
    --VQ-turnon $vqonarg --exp-dir $expdir \
    --data-train $data_tr --data-val $data_dt \
    $args >> "$expdir/train.out" 2>&1
