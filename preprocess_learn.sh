DATASET_DIR='../../dataset/filter_set1bin'
python="/home/$(whoami)/anaconda3/envs/tf/bin/python"
$python -m src.generate_test_data "${DATASET_DIR}/set1bin.valid.txt" "${DATASET_DIR}/valid.dat" 
$python -m src.generate_test_data "${DATASET_DIR}/set1bin.test.txt" "${DATASET_DIR}/test.dat" 
