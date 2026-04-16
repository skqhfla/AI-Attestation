#!/usr/bin/env sh

# -----------------------------------------------
# Script: train_CIFAR100.sh
# Usage: ./train_CIFAR100.sh <arch> <dataset>
# Example: ./train_CIFAR100.sh googlenet cifar100
# -----------------------------------------------


if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <arch> <dataset>"
    exit 1
fi

ARCH=$1
DATASET=$2

enable_tb_display=false # enable tensorboard display
epochs=1800
train_batch_size=128
test_batch_size=128
label_info=binarized

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

# Automatic check the host and configure
case $HOST in
"cpss-desktop")
    PYTHON="/home/cpss/anaconda3/envs/bfa310/bin/python3" # python environment path
    TENSORBOARD='/home/cpss/anaconda3/envs/bfa310/bin/tensorboard' # tensorboard environment path
    data_path='./data/cifar100' # dataset path
    ;;
esac

DATE=`date +%Y-%m-%d`
#DATE="2025-07-22"

if [ ! -d "$DIRECTORY" ]; then
    mkdir -p  ./save/${DATE}/
fi

save_path=./save/${DATE}/${DATASET}_${ARCH}_${epochs}_${label_info}
checkpoint=${save_path}/checkpoint.pth.tar
tb_path=${save_path}/tb_log  #tensorboard log path


############### Neural network ############################
{
$PYTHON main.py \
    --dataset ${DATASET} \
    --data_path ${data_path} \
    --arch ${ARCH} \
    --save_path ${save_path} \
    --epochs ${epochs} \
    --learning_rate 3e-4 \
    --schedule 80 120 \
    --gammas 0.1 0.1 \
    --attack_sample_size ${train_batch_size} \
    --test_batch_size ${test_batch_size} \
    --workers 4 \
    --ngpu 1 \
    --gpu_id 0 \
    --print_freq 100 \
    --decay 0.0005 \
    --momentum 0.9 \
    --resume ${checkpoint}  
#    --clustering --lambda_coeff 1e-3 \
}

############## Tensorboard logging ##########################
{
if [ "$enable_tb_display" = true ]; then 
    sleep 30 
    wait
    $TENSORBOARD --logdir $tb_path  --port=6006
fi
} &
{
if [ "$enable_tb_display" = true ]; then
    sleep 45
    wait
    case $HOST in
    "Hydrogen")
        firefox http://0.0.0.0:6006/
        ;;
    "alpha")
        google-chrome http://0.0.0.0:6006/
        ;;
    esac
fi 
} &
wait
