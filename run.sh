DATA_ROOT=/home/yuminghuang/dataset
STORAGE_ROOT=/home/yuminghuang/dataexp/NavigationField/OrField

TRAINING_SET=raw-derived-semantic-kitti; TRAINING_FEAT=example;
DEPLOY_SET=semantic-kitti; DEPLOY_SEQS="08,13,15,16,18,19"

# DATASET_PATH=${DATA_ROOT}/${TRAINING_SET}/sequences DATASET_TYPE=${TRAINING_SET} python train_net.py --num-gpus 1 --config-file configs/model.yaml OUTPUT_DIR ./output.${TRAINING_FEAT}
DEPLOY_SEQS=${DEPLOY_SEQS} DATASET_PATH=${DATA_ROOT}/${DEPLOY_SET}/sequences DATASET_TYPE=${DEPLOY_SET} python export_model.py \
--run-eval \
--format onnx \
--config-file ./output.${TRAINING_FEAT}/config.yaml \
--output ${STORAGE_ROOT}/output.${TRAINING_FEAT} \
MODEL.WEIGHTS ./output.${TRAINING_FEAT}/model_final.pth \
MODEL.DEVICE "cuda"
# PYTHONPATH=. DEPLOY_SEQS=${DEPLOY_SEQS} DATASET_PATH=${DATA_ROOT}/${DEPLOY_SET}/sequences OUTPUT_DIR=${STORAGE_ROOT}/output.${TRAINING_FEAT} python evaluator/latex.py