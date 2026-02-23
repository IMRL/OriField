DATA_ROOT=/home/yuminghuang/dataset

DATASET_PATH=${DATA_ROOT}/raw-derived-semantic-kitti/sequences DATASET_TYPE=raw-derived-semantic-kitti PREPROCESS_SCANS_TRAINING_ACC=temporal PREPROCESS_SCANS_TRAINING_CROP=none python prepare_training.py
DATASET_PATH=${DATA_ROOT}/semantic-kitti/sequences DATASET_TYPE=semantic-kitti python prepare_evaluation.py