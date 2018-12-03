#!/usr/bin/env bash

set -e

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  python="/home/$(whoami)/anaconda3/envs/tf/bin/python"
else
  python="/Users/ezail/anaconda3/bin/python"
fi

expt_dir="data/arxiv_obj/"
DATASET_DIR="../../dataset/arxiv"
dim="70"
max_rk="21"

# === Implicit Intervention ===
log_dir="${expt_dir}/log"
npy_dir="${expt_dir}/data"
res_dir="${expt_dir}/result"

$python -m src.generate_feat -m ${max_rk} --complete "$DATASET_DIR/queries_multi.tsv" \
  "$DATASET_DIR/clicks_multi.tsv" "${npy_dir}/train.feat.npy"

$python -m src.extract_click -m ${max_rk} --complete "$DATASET_DIR/queries_multi.tsv" \
  "$DATASET_DIR/clicks_multi.tsv" "${npy_dir}/train.click.npy"

# === mlp without relevance ==
echo 'Estimating without relevance model...'
model_dir="${res_dir}/mlp"
mkdir -p ${expt_dir}
$python -m src.model.ann_arxiv_complete -m ${max_rk} -d ${dim} -n1 128 \
  mlp ${npy_dir} ${expt_dir}

# === mlp with relevance ===
echo 'Estimating with relevance model...'
model_dir="${res_dir}/mlp_rel"
mkdir -p ${expt_dir}
$python -m src.model.ann_arxiv_complete -m ${max_rk} -d ${dim} -n1 32 -n2 32 \
  mlp ${npy_dir} ${expt_dir}
