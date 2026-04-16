#!/usr/bin/env bash

# -----------------------------------------
# Script: bash run_all_cifar100.sh
# Description: Train all CIFAR-100 model/optimizer combinations
# using train_CIFAR100.sh with arguments (arch, dataset, optimizer)
# -----------------------------------------

DATASET="cifar100"

MODELS=(
#    googlenet
#    googlenet_quan
    densenet121
    densenet121_quan
#    densenet169_quan
#    densenet201_quan
#    densenet161_quan
    shufflenetv2_quan
    squeezenet_quan
#    mobilenetv2_quan
)

OPTIMIZERS=(
    SGD
)

# 순차 실행 (또는 필요 시 병렬로 "&" 붙여 사용 가능)
for MODEL in "${MODELS[@]}"; do
    for OPTIMIZER in "${OPTIMIZERS[@]}"; do
        echo "Running: train_CIFAR100.sh ${MODEL} ${DATASET} ${OPTIMIZER}"
        ./train_CIFAR100.sh ${MODEL} ${DATASET} ${OPTIMIZER}
        echo " Finished: ${MODEL} with ${OPTIMIZER}"
        echo "----------------------------------------------"
    done
done

echo "All CIFAR-100 training jobs complete."
