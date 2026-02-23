DATA_ROOT=/home/yuminghuang/dataset
STORAGE_ROOT=/home/yuminghuang/dataexp/NavigationField/OrField

TRAINING_SET=um-osmplanner-4.bev_0.16_224.trainingcrop_1.livox; TRAINING_FEAT=example.um;
DEPLOY_SET=um-osmplanner-4.bev_0.16_224.trainingcrop_1.livox; DEPLOY_SEQS="scene 6,scene 7,scene 8"

# DATASET_PATH=${DATA_ROOT}/${TRAINING_SET} DATASET_TYPE=um-osm-planner python train_net.py --num-gpus 1 --config-file configs/model.yaml OUTPUT_DIR ./output.${TRAINING_FEAT}
DEPLOY_SEQS=${DEPLOY_SEQS} DATASET_PATH=${DATA_ROOT}/${DEPLOY_SET} DATASET_TYPE=um-osm-planner python export_model.py \
--run-eval \
--format onnx \
--config-file ./output.${TRAINING_FEAT}/config.yaml \
--output ${STORAGE_ROOT}/output.${TRAINING_FEAT} \
MODEL.WEIGHTS ./output.${TRAINING_FEAT}/model_final.pth \
MODEL.DEVICE "cuda"
# PYTHONPATH=. DEPLOY_SEQS=${DEPLOY_SEQS} DATASET_PATH=${DATA_ROOT}/${DEPLOY_SET} OUTPUT_DIR=${STORAGE_ROOT}/output.${TRAINING_FEAT} python evaluator/latex.py