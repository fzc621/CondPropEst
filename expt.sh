#!/usr/bin/env bash

set -e

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  python="/home/$(whoami)/anaconda3/envs/tf/bin/python"
else
  python="/Users/ezail/anaconda3/bin/python"
fi

if [[ $# == 3 ]]; then
  expt_dir="data/set1bin/expt/$1/$2/$3"
  DATA_DIR="data/set1bin/input/$1/$2/$3"
elif [[ $# == 2 ]]; then
  expt_dir="data/set1bin/expt/$1/$2"
  DATA_DIR="data/set1bin/input/$1/$2"
else
  expt_dir="data/set1bin/expt"
  DATA_DIR="data/set1bin/input"
fi

if [[ "$1" == "weight" ]]; then
  w="$2"
  sw="5"
  st="0"
elif [[ "$1" == "sweep" ]]; then
  sw="$2"
  w="0.1"
  st="0"
else
  sw="5"
  w="0.1"
  st="0"
fi

DATASET_DIR='../../dataset/filter_set1bin'
strength_dir="${expt_dir}/strength/${st}"
svm_dir="../svm_rank"
svm_learn="${svm_dir}/svm_rank_learn"
svm_classify="${svm_dir}/svm_rank_classify"
log_dir="${strength_dir}/log"
res_dir="${strength_dir}/result"
ground_truth_dir="${strength_dir}/ground_truth"
NPY_DIR="${strength_dir}/data"

echo 'Genrating two rankers'
mkdir -p ${DATA_DIR}
$python -m src.sample_slice -o 0.2 "${DATASET_DIR}/set1bin.train.txt" $DATA_DIR

mkdir -p ${expt_dir}
for i in 0 1;
do
  $svm_learn -c 3 "${DATA_DIR}/set1bin.slice${i}.txt" \
    "${expt_dir}/rank${i}.dat" > /dev/null
  $svm_classify "${DATA_DIR}/set1bin.train.txt" "${expt_dir}/rank${i}.dat" \
    "${expt_dir}/train.score${i}.dat" > /dev/null
done

echo 'Synthesizing features and calculating true propensity'
$python -m src.sim_feat -st ${st} "${DATA_DIR}/set1bin.train.txt" $DATA_DIR
$python -m src.sim_feat -st ${st} "${DATASET_DIR}/set1bin.valid.txt" $DATA_DIR
$python -m src.sim_feat -st ${st} "${DATASET_DIR}/set1bin.test.txt" $DATA_DIR

$python -m src.cal_prop -n 10 -d 699 -m power -w $w "${ground_truth_dir}/para.dat" "${DATA_DIR}/set1bin.train.feat.txt" \
  "${ground_truth_dir}/set1bin.train.prop.txt"
$python -m src.cal_prop -n 10 -d 699 -m power -w $w "${ground_truth_dir}/para.dat" "${DATA_DIR}/set1bin.valid.feat.txt" \
  "${ground_truth_dir}/set1bin.valid.prop.txt"
$python -m src.cal_prop -n 10 -d 699 -m power -w $w "${ground_truth_dir}/para.dat" "${DATA_DIR}/set1bin.test.feat.txt" \
  "${ground_truth_dir}/set1bin.test.prop.txt"

echo 'Start to generate click logs...'
for i in 0 1;
do
  $python -m src.sim_click -s $sw -m power -d 699 -w $w "${ground_truth_dir}/para.dat" \
    "${DATA_DIR}/set1bin.train.txt" "${expt_dir}/train.score${i}.dat" \
    "${DATA_DIR}/set1bin.train.feat.txt" "${log_dir}/train.log${i}.txt"
done

$python -m src.data_process -m 10 -d 699 train ${log_dir} ${DATA_DIR} ${NPY_DIR}
rm -rf ${DATA_DIR}

=== w/o cond ===
model_dir="${res_dir}/wo_cond"
mkdir -p ${model_dir}
echo 'Estimating without query feature'
$python -m src.model.wo_cond -n 10 --log_dir ${log_dir} --gt_dir ${ground_truth_dir} ${model_dir} > "${model_dir}/train.txt"
$python -m src.model.wo_cond --test --gt_dir ${ground_truth_dir} ${model_dir} > "${model_dir}/test.txt"

# === mlp without relevance ==
echo 'Estimating without relevance model...'
model_dir="${res_dir}/mlp"
mkdir -p ${model_dir}
$python -m src.model.ann -m 10 -d 699 -n1 1024 --gt_dir ${ground_truth_dir} \
  train mlp ${NPY_DIR} ${model_dir} #> "${model_dir}/train.txt"
$python -m src.model.ann -m 10 -d 699 -n1 1024 --gt_dir ${ground_truth_dir} \
  test mlp ${NPY_DIR} ${model_dir} #> "${model_dir}/test.txt"
#
# # === mlp with relevance ===
# echo 'Estimating with relevance model...'
# model_dir="${res_dir}/mlp_rel"
# mkdir -p ${model_dir}
# $python -m src.model.ann -m 10 -d 10 -n1 9 -n2 11 --gt_dir ${ground_truth_dir} \
#   train mlp ${NPY_DIR} ${model_dir} > "${model_dir}/train.txt"
# $python -m src.model.ann -m 10 -d 10 -n1 9 -n2 11 --gt_dir ${ground_truth_dir} \
#   valid mlp ${NPY_DIR} ${model_dir} > "${model_dir}/valid.txt"
# $python -m src.model.ann -m 10 -d 10 -n1 9 -n2 11 --gt_dir ${ground_truth_dir} \
#   test mlp ${NPY_DIR} ${model_dir} > "${model_dir}/test.txt"

# # === mlp without relevance ==
# echo 'Estimating without relevance model...'
# for n1 in $(seq 8 16)
# do
#   model_dir="${res_dir}/mlp/${n1}"
#   mkdir -p ${model_dir}
#   echo "N1 = ${n1}"
#   $python -m src.model.ann -m 10 -d 10 -n1 ${n1} --gt_dir ${ground_truth_dir} \
#     train mlp ${NPY_DIR} ${model_dir} > "${model_dir}/train.txt"
#   $python -m src.model.ann -m 10 -d 10 -n1 ${n1} --gt_dir ${ground_truth_dir} \
#     valid mlp ${NPY_DIR} ${model_dir} > "${model_dir}/valid.txt"
#   $python -m src.model.ann -m 10 -d 10 -n1 ${n1} --gt_dir ${ground_truth_dir} \
#     test mlp ${NPY_DIR} ${model_dir} > "${model_dir}/test.txt"
# done
#
# # === mlp with relevance ===
# echo 'Estimating with relevance model...'
# for n1 in $(seq 8 16)
# do
#   for n2 in $(seq 8 16)
#   do
#     model_dir="${res_dir}/mlp_rel/${n1}/${n2}"
#     mkdir -p ${model_dir}
#     echo "N1 = ${n1} N2 = ${n2}"
#     $python -m src.model.ann -m 10 -d 10 -n1 ${n1} -n2 ${n2} --gt_dir ${ground_truth_dir} \
#       train mlp ${NPY_DIR} ${model_dir} > "${model_dir}/train.txt"
#     $python -m src.model.ann -m 10 -d 10 -n1 ${n1} -n2 ${n2} --gt_dir ${ground_truth_dir} \
#       valid mlp ${NPY_DIR} ${model_dir} > "${model_dir}/valid.txt"
#     $python -m src.model.ann -m 10 -d 10 -n1 ${n1} -n2 ${n2} --gt_dir ${ground_truth_dir} \
#       test mlp ${NPY_DIR} ${model_dir} > "${model_dir}/test.txt"
#   done
# done
