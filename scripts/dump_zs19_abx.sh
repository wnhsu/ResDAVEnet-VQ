#!/bin/bash 

# Author: Wei-Ning Hsu


set -eu

ckpt_path=$1  # ./exps/pretrained/RDVQ_00000_01000
layer=$2      # quant2
out_path=$3   # ./exps/output_zs19/RDVQ_00000_01000_quant2.zip

# Note that we use DatasetFolder to find wav files, which assumes data to
# be store at `<dir>/<label>/*.wav`. Hence it is necessary to move the ZS19
# files from `.../english/test/*.wav` to `.../english/test_clean/test/*.wav`
zs_dir="/data/sls/temp/wnhsu/experiments/zs19_evaluation/shared/iclr20/data4pytorch/english/test_clean/"


tmp_dir=$(mktemp -d -t zs19-XXXXXX)
trap "echo removing temporary dir $tmp_dir && rm -rf $tmp_dir" EXIT INT TERM

python run_extract_feats_abx.py \
  --layer $layer --input_ext "wav" --output_name_level=1 \
  $ckpt_path $zs_dir "$tmp_dir/english/test"

mkdir -p $(dirname $out_path) && out_path=$(readlink -f $out_path)
cur_dir=$PWD && cd $tmp_dir && zip -qr $out_path ./* && cd $cur_dir

echo -n "Run \"bash evaluate.sh $out_path test dtw_cosine\" in the docker "
echo -n "provided by ZeroSpeech19 to compute ABX error and bit-rate"
echo ""
