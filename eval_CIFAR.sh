#!/usr/bin/env sh

################# NOTE #############
# This script is used for observation-x in paper

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <n_iter> <k_top> <dataset>"
    exit 1
fi

n_iter=$1 # number of iteration to perform BFA
k_top=$2 # only check k_top weights with top gradient ranking in each layer
dataset=$3


############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

# Automatic check the host and configure
case $HOST in
"cpss-desktop")
    PYTHON="/home/cpss/anaconda3/envs/bfa310/bin/python3" # python environment path
    TENSORBOARD='/home/cpss/anaconda3/envs/bfa310/bin/tensorboard' # tensorboard environment path
    data_path='./data/cifar10' # dataset path
    ;;
esac

DATE="2025-06-11"
#DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir -p ./save/${DATE}/
fi

############### Configurations ########################
enable_tb_display=false # enable tensorboard display
model=resnet20_quan
optimizer=SGD
epochs=160
label_info=binarized
test_batch_size=128

attack_sample_size=128 # number of data used for BFA



model_path=./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info}
save_path=${model_path}/new_exp
tb_path=${save_path}/tb_log  #tensorboard log path

# set the pretrained model path
#pretrained_model=/home/elliot/Documents/CVPR_2020/BFA_defense/BFA_defense/save/2019-11-12/cifar10_vanilla_resnet20_160_SGD_idx_1/model_best.pth.tar
pretrained_model=${model_path}/model_best.pth.tar
  #tensorboard log path

############### Neural network ############################
{
$PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ${save_path} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --test_batch_size ${test_batch_size} \
    --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --evaluate --resume ${pretrained_model} --fine_tune \
    --attack_sample_size ${attack_sample_size} \
    --reset_weight --bfa --n_iter ${n_iter} --k_top ${k_top} \
    
} &
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
