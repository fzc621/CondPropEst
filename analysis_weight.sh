#!/usr/bin/env bash

set -e

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  python="/home/$(whoami)/anaconda3/envs/tf/bin/python"
else
  python="/Users/ezail/anaconda3/bin/python"
fi

st='1'
DATASET_DIR='../../dataset/filter_set1bin'
DATA_DIR="data/set1bin/input"
dim="699"
strength_dir="${expt_dir}/strength/${st}"
feat_dir="$DATA_DIR/${st}"
expt_dir="data/set1bin/expt"

# Remember to reset sim_feat
mkdir -p $feat_dir
$python -m src.sim_feat -st ${st} -d ${dim} "${DATA_DIR}/set1bin.train.txt" $feat_dir

$python -m src.diagnostic "${DATA_DIR}/set1bin.train.txt" \
	"${expt_dir}/train.score0.dat" "${feat_dir}/set1bin.train.feat.txt"
