#!/usr/bin/env bash

# -----------------------------------------
# Script: bash run_all_train_cifar10.sh
# Description: Train all CIFAR-10 model/optimizer combinations
# using train_CIFAR.sh with arguments (arch, dataset, optimizer)
# -----------------------------------------

# Set dataset
DATASET="cifar10"

# CIFAR-10 대상 모델 목록
MODELS=(
    vanilla_resnet20
    resnet20_quan
    resnet20_bin
    vgg11
    vgg11_bn
    vgg11_quan
    vgg11_bn_quan
    vgg11_bn_bin
)

# 사용될 옵티마이저 목록
OPTIMIZERS=(
    SGD
)

# 순차 실행 (또는 필요 시 병렬로 "&" 붙여 사용 가능)
for MODEL in "${MODELS[@]}"; do
    for OPTIMIZER in "${OPTIMIZERS[@]}"; do
        echo "Running: train_CIFAR.sh ${MODEL} ${DATASET} ${OPTIMIZER}"
        ./train_CIFAR.sh ${MODEL} ${DATASET} ${OPTIMIZER}
        echo " Finished: ${MODEL} with ${OPTIMIZER}"
        echo "----------------------------------------------"
    done
done

echo "All CIFAR-10 training jobs complete."
