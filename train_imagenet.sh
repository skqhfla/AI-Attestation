#!/usr/bin/env sh

# -----------------------------------------------
# Script: train_imagenet.sh
# Usage: ./train_imagenet.sh <arch> <dataset> <optimizer>
# Example: ./train_imagenet.sh resnet18 imagenet SGD
# -----------------------------------------------


if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <arch> <dataset> <optimizer>"
    exit 1
fi

ARCH=$1
DATASET=$2
OPTIMIZER=$3

enable_tb_display=false # enable tensorboard display
epochs=50
train_batch_size=256
test_batch_size=256
label_info=idx_11

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

if [ ! -d "$DIRECTORY" ]; then
    mkdir -p ./save/${DATE}/
fi

############### Configurations ########################


save_path=./save/${DATE}/${DATASET}_${ARCH}_${epochs}_${OPTIMIZER}_${label_info}
tb_path=${save_path}/tb_log  #tensorboard log path

############### Neural network ############################
{
$PYTHON main.py --dataset ${DATASET} \
    --data_path ${data_path}   \
    --arch ${ARCH} \
    --save_path ${save_path} \
    --epochs ${epochs} 
    --learning_rate 0.001 \
    --optimizer ${OPTIMIZER} \
    --schedule 30 40 45 \ 
    --gammas 0.2 0.2 0.5 \
    --test_batch_size ${test_batch_size} \
    --attack_sample_size ${train_batch_size} \
    --workers 8 \
    --ngpu 1 \
    --gpu_id 0 \
    --print_freq 100 
    --decay 0.000005 \
    # --momentum 0.9 \
    # --evaluate
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
    "cpss-desktop")
        google-chrome http://0.0.0.0:6006/
        ;;
    esac
fi 
} &
wait
