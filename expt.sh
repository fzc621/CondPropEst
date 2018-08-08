set -e

DATASET_DIR='../../dataset/set1bin'
DATA_DIR='data/set1bin'

python -m src.sim_feat $DATASET_DIR $DATA_DIR
