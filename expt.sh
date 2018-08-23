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

# ground_truth_dir="${res_dir}/gt"
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

# === w/o cond ===
model_dir="${res_dir}/wo_cond"
python -m src.model.wo_cond -n 10 "${DATA_DIR}/set1bin.train.feat.txt" \
  ${log_dir} ${model_dir}

# === w/ cond logistic ===
# model_dir="${res_dir}/logistic"
# python -m src.model.logistic -n 10 -d 10 "${DATA_DIR}/set1bin.train.feat.txt" \
#   ${log_dir} ${model_dir}

# === w/ cond recover ===
# model_dir="${res_dir}/recover"
# if [[ "$OSTYPE" == "linux-gnu" ]]; then
#   ~/submit_job.pl "${PYTHON_PATH} -m src.model.recover -n 10 -d 10 '${DATA_DIR}/set1bin.train.feat.txt' ${log_dir} ${model_dir}"
# else
#   python -m src.model.recover -n 10 -d 10 '${DATA_DIR}/set1bin.train.feat.txt' \
#     ${log_dir} ${model_dir}
# fi
