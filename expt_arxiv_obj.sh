#!/usr/bin/env bash

set -e

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  python="/home/$(whoami)/anaconda3/envs/tf/bin/python"
else
  python="/Users/ezail/anaconda3/bin/python"
fi

expt_dir="data/arxiv_obj"
DATA_DIR="data/arxiv/input"
DATASET_DIR="../../dataset/arxiv"
max_rk="21"

# === Implicit Intervention ===
res_dir="${expt_dir}/result"
$python -m src.extract_click -m ${max_rk} "${DATA_DIR}/train_queries_multi.tsv" \
  "$DATASET_DIR/clicks_multi.tsv" "${DATA_DIR}/train.click.npy"
$python -m src.extract_click -m ${max_rk} "${DATA_DIR}/valid_queries_multi.tsv" \
  "$DATASET_DIR/clicks_multi.tsv" "${DATA_DIR}/valid.click.npy"
$python -m src.extract_click -m ${max_rk} "${DATA_DIR}/test_queries_multi.tsv" \
  "$DATASET_DIR/clicks_multi.tsv" "${DATA_DIR}/test.click.npy"

# === PBM ===
model_dir="${res_dir}/pbm"
mkdir -p ${model_dir}
$python -m src.arxiv_obj.pbm -m ${max_rk} "${DATA_DIR}/train.click.npy" \
  "${DATA_DIR}/test.click.npy" "${model_dir}/test_loss.txt"

ns="16 32 64 128"
# === CPBM: Full Features ===
model_dir="${res_dir}/full"
dim="70"
echo 'Full features'
$python -m src.generate_feat -m ${max_rk} --complete --complex --query_len \
  --session --num_results --result_proportion "${DATA_DIR}/train_queries_multi.tsv" \
  "$DATASET_DIR/clicks_multi.tsv" "${DATA_DIR}/train.full.feat.npy"
$python -m src.generate_feat -m ${max_rk} --complete --complex --query_len \
  --session --num_results --result_proportion "${DATA_DIR}/valid_queries_multi.tsv" \
  "$DATASET_DIR/clicks_multi.tsv" "${DATA_DIR}/valid.full.feat.npy"
$python -m src.generate_feat -m ${max_rk} --complete --complex --query_len \
  --session --num_results --result_proportion "${DATA_DIR}/test_queries_multi.tsv" \
  "$DATASET_DIR/clicks_multi.tsv" "${DATA_DIR}/test.full.feat.npy"

mkdir -p ${model_dir}
for n1 in ${ns}
do
  for n2 in ${ns}
  do
    mlp_dir="${model_dir}/${n1}_${n2}"
    $python -m src.arxiv_obj.cpbm -m ${max_rk} -d ${dim} \
       -n1 ${n1} -n2 ${n2} full "${DATA_DIR}" "${mlp_dir}" \
      &> "${model_dir}/${n1}_${n2}.log"
  done
done

wait
# === CPBM: Complex Features ===
model_dir="${res_dir}/complex"
dim="10"
echo 'complex features'
$python -m src.generate_feat -m ${max_rk} --complete --complex \
  "${DATA_DIR}/train_queries_multi.tsv" "$DATASET_DIR/clicks_multi.tsv" \
  "${DATA_DIR}/train.complex.feat.npy"
$python -m src.generate_feat -m ${max_rk} --complete --complex\
  "${DATA_DIR}/valid_queries_multi.tsv" "$DATASET_DIR/clicks_multi.tsv" \
  "${DATA_DIR}/valid.complex.feat.npy"
$python -m src.generate_feat -m ${max_rk} --complete --complex \
  "${DATA_DIR}/test_queries_multi.tsv" "$DATASET_DIR/clicks_multi.tsv" \
  "${DATA_DIR}/test.complex.feat.npy"

mkdir -p ${model_dir}
for n1 in ${ns}
do
  for n2 in ${ns}
  do
    mlp_dir="${model_dir}/${n1}_${n2}"
    $python -m src.arxiv_obj.cpbm -m ${max_rk} -d ${dim} \
       -n1 ${n1} -n2 ${n2} complex "${DATA_DIR}" "${mlp_dir}" \
      &> "${model_dir}/${n1}_${n2}.log"
  done
done

wait
# === CPBM: Query-Length Features ===
model_dir="${res_dir}/query_len"
dim="10"
echo 'query_len features'
$python -m src.generate_feat -m ${max_rk} --complete --query_len \
  "${DATA_DIR}/train_queries_multi.tsv" "$DATASET_DIR/clicks_multi.tsv" \
  "${DATA_DIR}/train.query_len.feat.npy"
$python -m src.generate_feat -m ${max_rk} --complete --query_len\
  "${DATA_DIR}/valid_queries_multi.tsv" "$DATASET_DIR/clicks_multi.tsv" \
  "${DATA_DIR}/valid.query_len.feat.npy"
$python -m src.generate_feat -m ${max_rk} --complete --query_len \
  "${DATA_DIR}/test_queries_multi.tsv" "$DATASET_DIR/clicks_multi.tsv" \
  "${DATA_DIR}/test.query_len.feat.npy"

mkdir -p ${model_dir}
for n1 in ${ns}
do
  for n2 in ${ns}
  do
    mlp_dir="${model_dir}/${n1}_${n2}"
    $python -m src.arxiv_obj.cpbm -m ${max_rk} -d ${dim} \
       -n1 ${n1} -n2 ${n2} query_len "${DATA_DIR}" "${mlp_dir}" \
      &> "${model_dir}/${n1}_${n2}.log"
  done
done

wait
# === CPBM: Session Features ===
model_dir="${res_dir}/session"
dim="5"
echo 'session features'
$python -m src.generate_feat -m ${max_rk} --complete --session \
  "${DATA_DIR}/train_queries_multi.tsv" "$DATASET_DIR/clicks_multi.tsv" \
  "${DATA_DIR}/train.session.feat.npy"
$python -m src.generate_feat -m ${max_rk} --complete --session\
  "${DATA_DIR}/valid_queries_multi.tsv" "$DATASET_DIR/clicks_multi.tsv" \
  "${DATA_DIR}/valid.session.feat.npy"
$python -m src.generate_feat -m ${max_rk} --complete --session \
  "${DATA_DIR}/test_queries_multi.tsv" "$DATASET_DIR/clicks_multi.tsv" \
  "${DATA_DIR}/test.session.feat.npy"

mkdir -p ${model_dir}
for n1 in ${ns}
do
  for n2 in ${ns}
  do
    mlp_dir="${model_dir}/${n1}_${n2}"
    $python -m src.arxiv_obj.cpbm -m ${max_rk} -d ${dim} \
       -n1 ${n1} -n2 ${n2} session "${DATA_DIR}" "${mlp_dir}" \
     &> "${model_dir}/${n1}_${n2}.log"
  done
done

wait
# === CPBM: Num_of_results Features ===
model_dir="${res_dir}/num_results"
dim="10"
echo 'num of results features'
$python -m src.generate_feat -m ${max_rk} --complete --num_results \
  "${DATA_DIR}/train_queries_multi.tsv" "$DATASET_DIR/clicks_multi.tsv" \
  "${DATA_DIR}/train.num_results.feat.npy"
$python -m src.generate_feat -m ${max_rk} --complete --num_results\
  "${DATA_DIR}/valid_queries_multi.tsv" "$DATASET_DIR/clicks_multi.tsv" \
  "${DATA_DIR}/valid.num_results.feat.npy"
$python -m src.generate_feat -m ${max_rk} --complete --num_results \
  "${DATA_DIR}/test_queries_multi.tsv" "$DATASET_DIR/clicks_multi.tsv" \
  "${DATA_DIR}/test.num_results.feat.npy"

mkdir -p ${model_dir}
for n1 in ${ns}
do
  for n2 in ${ns}
  do
    mlp_dir="${model_dir}/${n1}_${n2}"
    $python -m src.arxiv_obj.cpbm -m ${max_rk} -d ${dim} \
       -n1 ${n1} -n2 ${n2} num_results "${DATA_DIR}" "${mlp_dir}" \
     &> "${model_dir}/${n1}_${n2}.log"
  done
done

wait
# === CPBM: R Features ===
model_dir="${res_dir}/result_proportion"
dim="35"
echo 'result proportion features'
$python -m src.generate_feat -m ${max_rk} --complete --result_proportion \
  "${DATA_DIR}/train_queries_multi.tsv" "$DATASET_DIR/clicks_multi.tsv" \
  "${DATA_DIR}/train.result_proportion.feat.npy"
$python -m src.generate_feat -m ${max_rk} --complete --result_proportion\
  "${DATA_DIR}/valid_queries_multi.tsv" "$DATASET_DIR/clicks_multi.tsv" \
  "${DATA_DIR}/valid.result_proportion.feat.npy"
$python -m src.generate_feat -m ${max_rk} --complete --result_proportion \
  "${DATA_DIR}/test_queries_multi.tsv" "$DATASET_DIR/clicks_multi.tsv" \
  "${DATA_DIR}/test.result_proportion.feat.npy"

mkdir -p ${model_dir}
for n1 in ${ns}
do
  for n2 in ${ns}
  do
    mlp_dir="${model_dir}/${n1}_${n2}"
    $python -m src.arxiv_obj.cpbm -m ${max_rk} -d ${dim} \
       -n1 ${n1} -n2 ${n2} result_proportion "${DATA_DIR}" "${mlp_dir}" \
      &> "${model_dir}/${n1}_${n2}.log"
  done
done
