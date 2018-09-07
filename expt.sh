#!/usr/bin/env bash

set -e

func="comp"

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  python="/home/ys756/anaconda3/envs/fzc/bin/python"
else
  python="/Users/ezail/anaconda3/bin/python"
fi

if [[ $# == 3 ]]; then
  expt_dir="data/set1bin_${func}/expt/$1/$2/$3"
elif [[ $# == 2 ]]; then
  expt_dir="data/set1bin_${func}/expt/$1/$2"
else
  expt_dir="data/set1bin_${func}/expt"
fi

if [[ "$1" == "overlap" ]]; then
  ol="$2"
  sw="5"
elif [[ "$1" == "sweep" ]]; then
  ol="0.2"
  sw="$2"
else
  ol="0.2"
  sw="5"
fi

DATASET_DIR='../../dataset/set1bin'
DATA_DIR="data/set1bin_${func}/input"
svm_dir="../svm_rank"
svm_learn="${svm_dir}/svm_rank_learn"
svm_classify="${svm_dir}/svm_rank_classify"
log_dir="${expt_dir}/log"
res_dir="${expt_dir}/result"
ground_truth_dir="${expt_dir}/ground_truth_$func"
#
# $python -m src.sample_slice -o $ol "${DATASET_DIR}/set1bin.train.txt" $DATA_DIR
# $python -m src.sim_feat ${DATA_DIR}/set1bin.train.txt ${DATASET_DIR}/set1bin.test.txt $DATA_DIR
#
# === ground truth ===
# $python -m src.cal_prop -n 10 -m $func "${ground_truth_dir}/para.dat" "${DATA_DIR}/set1bin.train.feat.txt" \
#   "${ground_truth_dir}/set1bin.train.prop.txt"
# $python -m src.cal_prop -n 10 -m $func "${ground_truth_dir}/para.dat" "${DATA_DIR}/set1bin.test.feat.txt" \
#   "${ground_truth_dir}/set1bin.test.prop.txt"


# for i in 0 1;
# do
#   # $svm_learn -c 3 "${DATA_DIR}/set1bin.slice${i}.txt" "${expt_dir}/rank${i}.dat"
#   # $svm_classify "${DATA_DIR}/set1bin.train.txt" "${expt_dir}/rank${i}.dat" \
#   #     "${expt_dir}/score${i}.dat"
#   $python -m src.sim_click -s $sw -m $func "${expt_dir}/para.dat" \
#     "${DATA_DIR}/set1bin.train.txt" "${expt_dir}/score${i}.dat" \
#     "${DATA_DIR}/set1bin.train.feat.txt" "${log_dir}/log${i}.txt"
# done

NPY_DIR="${expt_dir}/data"
# $python -m src.data_process -m 10 -d 10 ${log_dir} ${DATA_DIR} ${NPY_DIR}

# === w/o cond ===
model_dir="${res_dir}/wo_cond"
# $python -m src.model.wo_cond -n 10 --log_dir ${log_dir} ${model_dir}
# $python -m src.model.wo_cond --test --gt_dir ${ground_truth_dir} ${model_dir}

# === recover ===
model_dir="${res_dir}/recover_$func"
# $python -m src.model.recover_$func -m 10 -d 10 ${NPY_DIR} ${model_dir}
# $python -m src.model.recover_$func --test --gt_dir ${ground_truth_dir} ${NPY_DIR} ${model_dir}

# === mlp ===
model_dir="${res_dir}/ann/mlp"
$python -m src.model.ann -m 10 -d 10 -e 300 mlp ${NPY_DIR} ${model_dir} --gt_dir ${ground_truth_dir}
$python -m src.model.ann --test --gt_dir ${ground_truth_dir} mlp \
	${NPY_DIR} ${model_dir}
