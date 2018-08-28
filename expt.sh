#!/usr/bin/env bash

set -e

DATASET_DIR='../../dataset/set1bin'
DATA_DIR='data/set1bin/input'
PYTHON_PATH='/home/ys756/anaconda3/envs/fzc/bin/python'
svm_dir="../svm_rank"
svm_learn="${svm_dir}/svm_rank_learn"
svm_classify="${svm_dir}/svm_rank_classify"
expt_dir="data/set1bin/expt"
log_dir="${expt_dir}/log"
res_dir="${expt_dir}/result"

# python -m src.sample_slice "${DATASET_DIR}/set1bin.train.txt" $DATA_DIR
# python -m src.sim_feat $DATA_DIR $DATA_DIR

# === ground truth ===
ground_truth_dir="${res_dir}/gt"
# python -m src.cal_prop -n 10 "${expt_dir}/para.dat" "${DATA_DIR}/set1bin.train.feat.txt" \
#   "${ground_truth_dir}/set1bin.train.prop.txt"
# python -m src.cal_prop -n 10 "${expt_dir}/para.dat" "${DATA_DIR}/set1bin.valid.feat.txt" \
#   "${ground_truth_dir}/set1bin.valid.prop.txt"
# python -m src.cal_prop -n 10 "${expt_dir}/para.dat" "${DATA_DIR}/set1bin.test.feat.txt" \
#   "${ground_truth_dir}/set1bin.test.prop.txt"


# for i in 0 1;
# do
#   $svm_learn -c 3 "${DATA_DIR}/set1bin.slice${i}.txt" "${expt_dir}/rank${i}.dat"
#   $svm_classify "${DATA_DIR}/set1bin.train.txt" "${expt_dir}/rank${i}.dat" \
#       "${expt_dir}/score${i}.dat"
#   python -m src.sim_click -s 5 "${expt_dir}/para.dat" "${DATA_DIR}/set1bin.train.txt" \
#     "${expt_dir}/score${i}.dat" "${DATA_DIR}/set1bin.train.feat.txt" \
#     "${log_dir}/log${i}.txt"
# done

NPY_DIR="${expt_dir}/data"
# python -m src.data_process -m 10 -d 10 ${log_dir} ${DATA_DIR} ${NPY_DIR}

# === w/o cond ===
model_dir="${res_dir}/wo_cond"
# python -m src.model.wo_cond -n 10 --log_dir ${log_dir} ${model_dir}
# python -m src.model.wo_cond --test --gt ${ground_truth_dir} ${model_dir}

# === w/ cond recover ===
model_dir="${res_dir}/recover"
# python -m src.model.recover -m 10 -d 10 \
#   ${NPY_DIR} ${model_dir}
# python -m src.model.recover --test --gt ${ground_truth_dir} ${NPY_DIR} ${model_dir}

# === w/ cond mlp ===
model_dir="${res_dir}/mlp"
# python -m src.model.mlp -m 10 -d 10 \
#   ${NPY_DIR} ${model_dir}
# python -m src.model.recover --test --gt ${ground_truth_dir}/set1bin.test.prop.txt \
#   ${DATA_DIR}/set1bin.test.feat.txt ${model_dir}
