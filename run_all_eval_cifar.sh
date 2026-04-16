#!/usr/bin/env bash

# -----------------------------------------
# Script: bash run_all_eval_cifar10.sh
# Description: Train all CIFAR-10 model/optimizer combinations
# using train_CIFAR.sh with arguments (arch, dataset, optimizer)
# -----------------------------------------

# Set dataset
DATASET="cifar10"

 number of iteration to perform BFA
n_iter=(
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
)

# 사용될 옵티마이저 목록
k_top=(
    1
    3
    5
)

# 순차 실행 (또는 필요 시 병렬로 "&" 붙여 사용 가능)
for n in "${n_iter[@]}"; do
    for k in "${k_top[@]}"; do
        echo "Running: eval_CIFAR.sh ${n} ${k} ${DATASET}"
        ./eval_CIFAR.sh ${n} ${k} ${DATASET}
        echo " Finished: ${n} with ${k}"
        echo "----------------------------------------------"
    done
done

echo "All CIFAR-10 evaluation jobs complete."
