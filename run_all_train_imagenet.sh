#!/usr/bin/env bash

# -----------------------------------------
# Script: bash run_all_train_imagenet.sh
# Description: Train all ImageNet model/optimizer combinations
# using train_imagenet.sh with arguments (arch, dataset, optimizer)
# -----------------------------------------

DATASET="imagenet"

MODELS=(
    resnet18
    resnet18_quan
#    resnet34_quan
#    mobilenet_v2
#    mobilenet_v2_quan
    alexnet_quan
)

OPTIMIZERS=(
    Adam
)

# 순차 실행 (또는 필요 시 병렬로 "&" 붙여 사용 가능)
for MODEL in "${MODELS[@]}"; do
    for OPTIMIZER in "${OPTIMIZERS[@]}"; do
        echo "Running: train_imagenet.sh ${MODEL} ${DATASET} ${OPTIMIZER}"
        ./train_imagenet.sh ${MODEL} ${DATASET} ${OPTIMIZER}
        echo " Finished: ${MODEL} with ${OPTIMIZER}"
        echo "----------------------------------------------"
    done
done

echo "All imagenet training jobs complete."
