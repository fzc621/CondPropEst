#!/usr/bin/env bash

set -e

default_st="1"

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
  sts=${default_st}
elif [[ "$1" == "sweep" ]]; then
  sw="$2"
  w="0.2"
  sts=${default_st}
elif [[ "$1" == "strength" ]]; then
  sw="$2"
  w="0.2"
  sts="0 0.2 0.4 0.6 0.8 1.0"
else
  sw="5"
  w="0.2"
  sts=${default_st}
fi

DATASET_DIR='../../dataset/filter_set1bin'
svm_dir="../svm_rank"
svm_learn="${svm_dir}/svm_rank_learn"
svm_classify="${svm_dir}/svm_rank_classify"
prop_svm_dir="../svm_proprank"
prop_svm_learn="${prop_svm_dir}/svm_proprank_learn"
prop_svm_classify="${prop_svm_dir}/svm_proprank_classify"
dim="10"


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

for st in $sts; do
  if [[ $sts == ${default_st} ]]; then
    strength_dir="${expt_dir}"
    feat_dir="$DATA_DIR"
  else
    strength_dir="${expt_dir}/strength/${st}"
    feat_dir="$DATA_DIR/${st}"
  fi

  log_dir="${strength_dir}/log"
  res_dir="${strength_dir}/result"
  ground_truth_dir="${strength_dir}/ground_truth"
  NPY_DIR="${strength_dir}/data"

  echo 'Synthesizing features and calculating true propensity'
  mkdir -p $feat_dir
  $python -m src.sim_feat -st ${st} -d ${dim} "${DATA_DIR}/set1bin.train.txt" $feat_dir
  $python -m src.sim_feat -st ${st} -d ${dim} "${DATASET_DIR}/set1bin.valid.txt" $feat_dir
  $python -m src.sim_feat -st ${st} -d ${dim} "${DATASET_DIR}/set1bin.test.txt" $feat_dir

  $python -m src.cal_prop -n 10 -d ${dim} -w $w "${expt_dir}/para.dat" "${feat_dir}/set1bin.train.feat.txt" \
    "${ground_truth_dir}/set1bin.train.prop.txt"
  $python -m src.cal_prop -n 10 -d ${dim} -w $w "${expt_dir}/para.dat" "${feat_dir}/set1bin.valid.feat.txt" \
    "${ground_truth_dir}/set1bin.valid.prop.txt"
  $python -m src.cal_prop -n 10 -d ${dim} -w $w "${expt_dir}/para.dat" "${feat_dir}/set1bin.test.feat.txt" \
    "${ground_truth_dir}/set1bin.test.prop.txt"

  echo 'Start to generate click logs...'
  for i in 0 1;
  do
    $python -m src.sim_click -s $sw -m power -d ${dim} -w $w "${expt_dir}/para.dat" \
      "${DATA_DIR}/set1bin.train.txt" "${expt_dir}/train.score${i}.dat" \
      "${feat_dir}/set1bin.train.feat.txt" "${log_dir}/train.log${i}.txt"
  done

  $python -m src.data_process -m 10 -d ${dim} train ${log_dir} ${feat_dir} ${NPY_DIR}

  # === PBM ===
  model_dir="${res_dir}/pbm"
  mkdir -p ${model_dir}
  echo 'Estimating without query feature'
  $python -m src.model.pbm -n 10 --log_dir ${log_dir} --gt_dir ${ground_truth_dir} ${model_dir} > "${model_dir}/train.txt"
  # $python -m src.model.wo_cond --test --gt_dir ${ground_truth_dir} ${model_dir} > "${model_dir}/test.txt"

  # === CPBM without relevance ==
  # echo 'Estimating without relevance model...'
  # model_dir="${res_dir}/mlp"
  # mkdir -p ${model_dir}
  # $python -m src.model.ann -m 10 -d ${dim} -n1 16 --gt_dir ${ground_truth_dir} \
  #   train mlp ${NPY_DIR} ${model_dir} > "${model_dir}/train.txt"
  # $python -m src.model.ann -m 10 -d ${dim} -n1 16 --gt_dir ${ground_truth_dir} \
  #   test mlp ${NPY_DIR} ${model_dir} > "${model_dir}/test.txt"

  # === CPBM ===
  echo 'Estimating with relevance model...'
  model_dir="${res_dir}/cpbm"
  mkdir -p ${model_dir}
  $python -m src.model.ann -m 10 -d ${dim} -n1 9 -n2 11 --gt_dir ${ground_truth_dir} \
    train mlp_rel ${NPY_DIR} ${model_dir} > "${model_dir}/train.txt"
  # $python -m src.model.ann -m 10 -d ${dim} -n1 9 -n2 11 --gt_dir ${ground_truth_dir} \
  #   valid mlp_rel ${NPY_DIR} ${model_dir} > "${model_dir}/valid.txt"
  # $python -m src.model.ann -m 10 -d ${dim} -n1 9 -n2 11 --gt_dir ${ground_truth_dir} \
  #   test mlp_rel ${NPY_DIR} ${model_dir} > "${model_dir}/test.txt"

  # # === mlp without relevance ==
  # echo 'Estimating without relevance model...'
  # for n1 in $(seq 8 16)
  # do
  #   model_dir="${res_dir}/mlp/${n1}"
  #   mkdir -p ${model_dir}
  #   echo "N1 = ${n1}"
  #   $python -m src.model.ann -m 10 -d ${dim} -n1 ${n1} --gt_dir ${ground_truth_dir} \
  #     train mlp ${NPY_DIR} ${model_dir} > "${model_dir}/train.txt"
  #   $python -m src.model.ann -m 10 -d ${dim} -n1 ${n1} --gt_dir ${ground_truth_dir} \
  #     valid mlp ${NPY_DIR} ${model_dir} > "${model_dir}/valid.txt"
  #   $python -m src.model.ann -m 10 -d ${dim} -n1 ${n1} --gt_dir ${ground_truth_dir} \
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

  # === Learning Part ===
  learn_dir="${expt_dir}/learn"
  ts="0.01 0.03 0.1 0.3"
  cs="0.1 0.3 1 3 10 30 100"
  mkdir -p ${learn_dir}
  # === PBM ===
  model_dir="${res_dir}/pbm"
  for t in ${ts}
  do
    $python -m src.generate_train_data -t ${t} "${model_dir}/set1bin.train.prop.txt" \
    	"${DATA_DIR}/set1bin.train.txt" "${expt_dir}/train.score0.dat" \
    	"${log_dir}/train.log0.txt" "${learn_dir}/pbm_train_t${t}.dat" &
  done

  # === CPBM ===
  model_dir="${res_dir}/cpbm"
  for t in ${ts}
  do
    $python -m src.generate_train_data -t ${t} "${model_dir}/set1bin.train.prop.txt" \
    	"${DATA_DIR}/set1bin.train.txt" "${expt_dir}/train.score0.dat" \
    	"${log_dir}/train.log0.txt" "${learn_dir}/cpbm_train_t${t}.dat" &
  done

  # === True propensity ===
  model_dir=${ground_truth_dir}
  for t in ${ts}
  do
    $python -m src.generate_train_data -t ${t} "${model_dir}/set1bin.train.prop.txt" \
    	"${DATA_DIR}/set1bin.train.txt" "${expt_dir}/train.score0.dat" \
    	"${log_dir}/train.log0.txt" "${learn_dir}/gt_train_t${t}.dat" &
  done

  $python -m src.generate_test_data "${DATASET_DIR}/set1bin.test.txt" "${learn_dir}/test.dat" &

  wait

  for model in "pbm cpbm gt"
  do
    for t in ${ts}
    do
      for c in ${cs}
      do
          ${prop_svm_learn} -c $C "${learn_dir}/${model}_train_t${t}.dat" "${learn_dir}/${model}_t${t}_c${c}.model" &> /dev/null &
      done
    done
  done

  wait

  for model in "pbm cpbm gt"
  do
    for t in ${ts}
    do
      for c in ${cs}
      do
          ${prop_svm_classify} "${learn_dir}/test.dat" "${learn_dir}/${model}_t${t}_c${c}.model" | grep SNIPS &> "${model}_t${t}_c${c}.log" &
      done
    done
  done
done
