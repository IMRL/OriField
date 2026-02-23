DATA_ROOT=/home/yuminghuang/dataset
export DATASET_PATH=${DATA_ROOT}/um-osmplanner-4.bev_0.16_224.trainingcrop_1

DATASET_TYPE=um-osm-planner PREPROCESS_SCANS_TRAINING_ACC=mapping PREPROCESS_SCANS_TRAINING_CROP=height python prepare_training.py
DATASET_TYPE=um-osm-planner python prepare_evaluation.py