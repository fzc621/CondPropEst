#!/usr/bin/env bash

set -e


if [[ "$OSTYPE" == "linux-gnu" ]]; then
  python="/home/ys756/anaconda3/envs/fzc/bin/python"
else
  python="/Users/ezail/anaconda3/bin/python"
fi

if [[ $# == 4 ]]; then
  func="$1"
  expt_dir="data/set1bin_${func}/expt/$2/$3/$4"
  DATA_DIR="data/set1bin_${func}/input/$2/$3/$4"
elif [[ $# == 3 ]]; then
  func="$1"
  expt_dir="data/set1bin_${func}/expt/$2/$3"
  DATA_DIR="data/set1bin_${func}/input/$2/$3"
else
  func='power'
  expt_dir="data/set1bin_${func}/expt"
  DATA_DIR="data/set1bin_${func}/input"
fi


if [[ "$2" == "weight" ]]; then
  w="$3"
  sw="5"
elif [[ "$2" == "sweep" ]]; then
  sw="$3"
  w="0.1"
else
  sw="5"
  w="0.1"
fi

DATASET_DIR='../../dataset/set1bin'

svm_dir="../svm_rank"
svm_learn="${svm_dir}/svm_rank_learn"
svm_classify="${svm_dir}/svm_rank_classify"
log_dir="${expt_dir}/log"
res_dir="${expt_dir}/result"
ground_truth_dir="${expt_dir}/ground_truth_$func"
NPY_DIR="${expt_dir}/data"

# mkdir -p ${DATA_DIR}
# cp ${DATASET_DIR}/set1bin.test.txt ${DATA_DIR}
# $python -m src.sample_slice -o 0.2 "${DATASET_DIR}/set1bin.train.txt" $DATA_DIR
#
# mkdir -p ${expt_dir}
# for i in 0 1;
# do
#   $svm_learn -c 3 "${DATA_DIR}/set1bin.slice${i}.txt" "${expt_dir}/rank${i}.dat"
#   $svm_classify "${DATA_DIR}/set1bin.train.txt" "${expt_dir}/rank${i}.dat" \
#       "${expt_dir}/train.score${i}.dat"
#   $svm_classify "${DATA_DIR}/set1bin.test.txt" "${expt_dir}/rank${i}.dat" \
#       "${expt_dir}/test.score${i}.dat"
# done
#
# $python -m src.sim_feat "${expt_dir}/train.score0.dat" "${expt_dir}/train.score1.dat" "${DATA_DIR}/set1bin.train.txt" $DATA_DIR
# $python -m src.sim_feat "${expt_dir}/test.score0.dat" "${expt_dir}/test.score1.dat" "${DATA_DIR}/set1bin.test.txt" $DATA_DIR
#
# $python -m src.cal_prop -n 10 -d 10 -m $func -w $w "${ground_truth_dir}/para.dat" "${DATA_DIR}/set1bin.train.feat.txt" \
#   "${ground_truth_dir}/set1bin.train.prop.txt"
# $python -m src.cal_prop -n 10 -d 10 -m $func -w $w "${ground_truth_dir}/para.dat" "${DATA_DIR}/set1bin.test.feat.txt" \
#   "${ground_truth_dir}/set1bin.test.prop.txt"
#
# echo 'Start to generate click logs...'
# for i in 0 1;
# do
#   $python -m src.sim_click -s $sw -m $func -d 10 -w $w "${ground_truth_dir}/para.dat" \
#     "${DATA_DIR}/set1bin.train.txt" "${expt_dir}/train.score${i}.dat" \
#     "${DATA_DIR}/set1bin.train.feat.txt" "${log_dir}/log${i}.txt"
# done
#
# $python -m src.data_process -m 10 -d 10 ${log_dir} ${DATA_DIR} ${NPY_DIR}
#
# # === w/o cond ===
# model_dir="${res_dir}/wo_cond"
# mkdir -p ${model_dir}
# echo 'Estimating without query feature'
# $python -m src.model.wo_cond -n 10 --log_dir ${log_dir} --gt_dir ${ground_truth_dir} ${model_dir} > "${model_dir}/train.txt"
# $python -m src.model.wo_cond --test --gt_dir ${ground_truth_dir} ${model_dir} > "${model_dir}/test.txt"

#
# === recover ===
model_dir="${res_dir}/recover_$func"
# mkdir -p ${model_dir}
# echo 'Recovering...'
# $python -m src.model.recover_$func -m 10 -d 10 --gt_dir ${ground_truth_dir} ${NPY_DIR} ${model_dir} > "${model_dir}/train.txt"
# $python -m src.model.recover_$func --test --gt_dir ${ground_truth_dir} ${NPY_DIR} ${model_dir} > "${model_dir}/test.txt"

# === mlp ===
model_dir="${res_dir}/ann/mlp/$1"
mkdir -p ${model_dir}
echo "Estimating with Multilayer Perception (N1 = $1)..."
$python -m src.model.ann -m 10 -d 10 -e 500 -n $1 mlp ${NPY_DIR} ${model_dir} --gt_dir ${ground_truth_dir} > "${model_dir}/train.txt"
$python -m src.model.ann -n $1 --test --gt_dir ${ground_truth_dir} mlp \
	${NPY_DIR} ${model_dir} > "${model_dir}/test.txt"

# === mlp with relevance model ===
model_dir="${res_dir}/ann/mlp_rel"
# mkdir -p ${model_dir}
# echo 'Estimating with Multilayer Perception and Relevance Model...'
# $python -m src.model.ann -m 10 -d 10 -e 500 mlp_rel ${NPY_DIR} ${model_dir} --gt_dir ${ground_truth_dir}  > "${model_dir}/train.txt"
# $python -m src.model.ann --test --gt_dir ${ground_truth_dir} mlp_rel \
# 	${NPY_DIR} ${model_dir} > "${model_dir}/test.txt"

# # === mlp best func ===
# model_dir="${res_dir}/ann/mlp_${func}_best"
# mkdir -p ${model_dir}
# echo 'Estimating with Best Multilayer Perception...'
# $python -m src.model.ann -m 10 -d 10 -e 500 mlp_${func}_best ${NPY_DIR} ${model_dir} --gt_dir ${ground_truth_dir} > "${model_dir}/train.txt"
# $python -m src.model.ann --test --gt_dir ${ground_truth_dir} mlp_${func}_best \
# 	${NPY_DIR} ${model_dir} > "${model_dir}/test.txt"
#
# # === best mlp with relevance model ===
# model_dir="${res_dir}/ann/mlp_best_rel"
# mkdir -p ${model_dir}
# echo 'Estimating with Multilayer Perception and the best Relevance Model...'
# $python -m src.model.ann -m 10 -d 10 -e 500 mlp_rel_best ${NPY_DIR} ${model_dir} --gt_dir ${ground_truth_dir} > "${model_dir}/train.txt"
# $python -m src.model.ann --test --gt_dir ${ground_truth_dir} mlp_rel_best \
# 	${NPY_DIR} ${model_dir} > "${model_dir}/test.txt"
