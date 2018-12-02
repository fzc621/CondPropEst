#!/usr/bin/env bash

set -e

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  python="/home/$(whoami)/anaconda3/envs/tf/bin/python"
else
  python="/Users/ezail/anaconda3/bin/python"
fi

expt_dir="data/arxiv/expt"
DATA_DIR="data/arxiv/input"
DATASET_DIR="../../dataset/arxiv"
dim="1"
max_rk="21"

# === Swap Intervention ===
model_dir="${expt_dir}/swap"
$python -m src.split_arxiv "${DATASET_DIR}/queries_multi_swap.tsv" "${model_dir}/input"

$python -m src.swap_prop -m ${max_rk} "${model_dir}/input/complex_queries_multi_swap.tsv" \
  "$DATASET_DIR/clicks_multi_swap.tsv" "${model_dir}/result/complex_prop.txt"
$python -m src.swap_prop -m ${max_rk}  "${model_dir}/input/simple_queries_multi_swap.tsv" \
  "$DATASET_DIR/clicks_multi_swap.tsv" "${model_dir}/result/simple_prop.txt"

# === Implicit Intervention ===
model_dir="${expt_dir}/allpairs"
feat_dir="${model_dir}/input"
log_dir="${model_dir}/log"
npy_dir="${model_dir}/data"
res_dir="${model_dir}/result"
$python -m src.extract_log -m ${max_rk} "$DATASET_DIR/queries_multi.tsv" \
  "$DATASET_DIR/clicks_multi.tsv" "${npy_dir}/train.feat.npy" "${npy_dir}/train.click.npy"

# === mlp without relevance ==
# echo 'Estimating without relevance model...'
# model_dir="${res_dir}/mlp"
# mkdir -p ${model_dir}
# $python -m src.model.ann_arxiv -m ${max_rk} -d ${dim} -n1 8 \
#   mlp ${npy_dir} ${model_dir}

# === mlp with relevance ===
echo 'Estimating with relevance model...'
model_dir="${res_dir}/mlp_rel"
mkdir -p ${model_dir}
$python -m src.model.ann_arxiv -m ${max_rk} -d ${dim} -n1 32 -n2 32 \
  mlp ${npy_dir} ${model_dir}
