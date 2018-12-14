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
$python -m src.arxiv_match.split_arxiv "${DATASET_DIR}/queries_multi_swap.tsv" "${DATA_DIR}"

$python -m src.arxiv_match.swap_prop -m ${max_rk} "${DATA_DIR}/complex_queries_multi_swap.tsv" \
  "$DATASET_DIR/clicks_multi_swap.tsv" "${model_dir}/result/complex_prop.txt"
$python -m src.arxiv_match.swap_prop -m ${max_rk}  "${DATA_DIR}/simple_queries_multi_swap.tsv" \
  "$DATASET_DIR/clicks_multi_swap.tsv" "${model_dir}/result/simple_prop.txt"

$python -m src.arxiv_match.bootstrap_swap -m ${max_rk} "${DATA_DIR}/complex_queries_multi_swap.tsv" \
  "$DATASET_DIR/clicks_multi_swap.tsv" "${model_dir}/result/complex_bootstrap.txt"
$python -m src.arxiv_match.bootstrap_swap -m ${max_rk} "${DATA_DIR}/simple_queries_multi_swap.tsv" \
  "$DATASET_DIR/clicks_multi_swap.tsv" "${model_dir}/result/simple_bootstrap.txt"

# === PBM ===
model_dir="${expt_dir}/pbm"
$python -m src.arxiv_match.split_arxiv "${DATASET_DIR}/queries_multi.tsv" "${DATA_DIR}"

$python -m src.arxiv_match.pbm_prop -m ${max_rk} "${DATA_DIR}/complex_queries_multi.tsv" \
  "$DATASET_DIR/clicks_multi.tsv" "${model_dir}/result/complex_prop.txt"
$python -m src.arxiv_match.pbm_prop -m ${max_rk} "${DATA_DIR}/simple_queries_multi.tsv" \
  "$DATASET_DIR/clicks_multi.tsv" "${model_dir}/result/simple_prop.txt"

$python -m src.arxiv_match.bootstrap_pbm -m ${max_rk} "${DATA_DIR}/complex_queries_multi.tsv" \
  "$DATASET_DIR/clicks_multi.tsv" "${model_dir}/result/complex_bootstrap.txt"
$python -m src.arxiv_match.bootstrap_pbm -m ${max_rk} "${DATA_DIR}/simple_queries_multi.tsv" \
  "$DATASET_DIR/clicks_multi.tsv" "${model_dir}/result/simple_bootstrap.txt"


# === CPBM ===
model_dir="${expt_dir}/cpbm"
feat_dir="${model_dir}/input"
npy_dir="${model_dir}/data"
res_dir="${model_dir}/result"
# $python -m src.generate_feat -m ${max_rk} "$DATASET_DIR/queries_multi.tsv" \
  # "$DATASET_DIR/clicks_multi.tsv" "${npy_dir}/train.feat.npy"

# $python -m src.extract_click -m ${max_rk} "$DATASET_DIR/queries_multi.tsv" \
#   "$DATASET_DIR/clicks_multi.tsv" "${DATA_DIR}/train.click.npy"

# echo 'Estimating with relevance model...'
# $python -m src.model.ann_arxiv -m ${max_rk} -d ${dim} -n1 2 -n2 2 \
#   mlp ${npy_dir} "${model_dir}/mlp"
