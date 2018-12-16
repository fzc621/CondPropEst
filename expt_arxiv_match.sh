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

# $python -m src.arxiv_match.split_data "${DATASET_DIR}/queries_multi.tsv" "${DATA_DIR}"

# === Swap Intervention ===
# echo '=== Swap Intervention ==='
# model_dir="${expt_dir}/swap"
# $python -m src.arxiv_match.split_arxiv "${DATASET_DIR}/queries_multi_swap.tsv" "${DATA_DIR}"
#
# $python -m src.arxiv_match.swap_prop -m ${max_rk} "${DATA_DIR}/complex_queries_multi_swap.tsv" \
#   "$DATASET_DIR/clicks_multi_swap.tsv" "${model_dir}/result/complex_prop.txt"
# $python -m src.arxiv_match.swap_prop -m ${max_rk}  "${DATA_DIR}/simple_queries_multi_swap.tsv" \
#   "$DATASET_DIR/clicks_multi_swap.tsv" "${model_dir}/result/simple_prop.txt"
#
# $python -m src.arxiv_match.bootstrap_swap -m ${max_rk} "${DATA_DIR}/complex_queries_multi_swap.tsv" \
#   "$DATASET_DIR/clicks_multi_swap.tsv" "${model_dir}/result/complex_bootstrap.txt"
# $python -m src.arxiv_match.bootstrap_swap -m ${max_rk} "${DATA_DIR}/simple_queries_multi_swap.tsv" \
#   "$DATASET_DIR/clicks_multi_swap.tsv" "${model_dir}/result/simple_bootstrap.txt"

# $python -m src.arxiv_match.plot_complex "${model_dir}/result" "${expt_dir}/swap.pdf"
# === PBM ===
# echo '=== PBM ==='
model_dir="${expt_dir}/pbm"
# $python -m src.arxiv_match.split_arxiv "${DATA_DIR}/train_queries_multi.tsv" "${DATA_DIR}"
#
# $python -m src.arxiv_match.pbm_prop -m ${max_rk} "${DATA_DIR}/complex_train_queries_multi.tsv" \
#   "$DATASET_DIR/clicks_multi.tsv" "${model_dir}/result/complex_prop.txt"
# $python -m src.arxiv_match.pbm_prop -m ${max_rk} "${DATA_DIR}/simple_train_queries_multi.tsv" \
#   "$DATASET_DIR/clicks_multi.tsv" "${model_dir}/result/simple_prop.txt"
#
# $python -m src.arxiv_match.bootstrap_pbm -m ${max_rk} "${DATA_DIR}/complex_train_queries_multi.tsv" \
#   "$DATASET_DIR/clicks_multi.tsv" "${model_dir}/result/complex_bootstrap.txt"
# $python -m src.arxiv_match.bootstrap_pbm -m ${max_rk} "${DATA_DIR}/simple_train_queries_multi.tsv" \
#   "$DATASET_DIR/clicks_multi.tsv" "${model_dir}/result/simple_bootstrap.txt"

# $python -m src.arxiv_match.plot_complex "${model_dir}/result" "${expt_dir}/pbm.pdf"
# === CPBM ===
echo '=== CPBM ==='
model_dir="${expt_dir}/cpbm"
mlp_dir="${model_dir}/mlp"
# $python -m src.generate_feat -m ${max_rk} "${DATA_DIR}/train_queries_multi.tsv" \
#   "$DATASET_DIR/clicks_multi.tsv" "${DATA_DIR}/train.feat.npy"
# $python -m src.generate_feat -m ${max_rk} "${DATA_DIR}/valid_queries_multi.tsv" \
#   "$DATASET_DIR/clicks_multi.tsv" "${DATA_DIR}/valid.feat.npy"
#
# $python -m src.extract_click -m ${max_rk} "${DATA_DIR}/train_queries_multi.tsv" \
#   "$DATASET_DIR/clicks_multi.tsv" "${DATA_DIR}/train.click.npy"
# $python -m src.extract_click -m ${max_rk} "${DATA_DIR}/valid_queries_multi.tsv" \
#   "$DATASET_DIR/clicks_multi.tsv" "${DATA_DIR}/valid.click.npy"

# mkdir -p ${mlp_dir}
# ns="4 8 16 32"
# for n1 in ${ns}
# do
#   for n2 in ${ns}
#   do
#     $python -m src.arxiv_match.cpbm_prop -m ${max_rk} -d ${dim} \
#        -n1 ${n1} -n2 ${n2} train mlp "${DATA_DIR}" \
#        "${mlp_dir}/${n1}_${n2}" &> "${mlp_dir}/train_${n1}_${n2}.log"&
#   done
# done
#
# wait
#
# for n1 in ${ns}
# do
#   for n2 in ${ns}
#   do
#     $python -m src.arxiv_match.cpbm_prop -m ${max_rk} -d ${dim} \
#       -n1 ${n1} -n2 ${n2} valid mlp "${DATA_DIR}" \
#       "${mlp_dir}/${n1}_${n2}" &> "${mlp_dir}/valid_${n1}_${n2}.log" &
#   done
# done

for i in $(seq 0 999)
do
  $python -m src.arxiv_match.bootstrap_cpbm -m ${max_rk} "${DATA_DIR}" \
    "${mlp_dir}" "${model_dir}/result/${i}"
done
