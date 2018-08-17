set -e

DATASET_DIR='../../dataset/set1bin'
DATA_DIR='data/set1bin/input'

svm_dir="../svm_rank"
svm_learn="${svm_dir}/svm_rank_learn"
svm_classify="${svm_dir}/svm_rank_classify"
expt_dir="data/set1bin/expt"
log_dir="${expt_dir}/log"
res_dir="${expt_dir}/result"

# python -m src.sample_slice "${DATASET_DIR}/set1bin.train.txt" $DATA_DIR
# python -m src.sim_feat $DATA_DIR $DATA_DIR

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
python -m src.model.wo_cond -n 10 "${DATA_DIR}/set1bin.train.feat.txt" \
  ${log_dir} "${res_dir}/wo_cond.txt"
