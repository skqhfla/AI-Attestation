#!/usr/bin/env bash


############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

# Automatic check the host and configure
case $HOST in
"cpss-desktop")
    PYTHON="/home/cpss/anaconda3/envs/bfa36/bin/python3" # python environment path
    TENSORBOARD='/home/cpss/anaconda3/envs/bfa310/bin/tensorboard' # tensorboard environment path
    data_path='./data/cifar' # dataset path
    ;;
esac

DATE="2025-08-19"
#DATE=`date +%Y-%m-%d`


if [ ! -d "$DIRECTORY" ]; then
    mkdir -p ./save/${DATE}/
fi

############### Configurations ########################
enable_tb_display=false # enable tensorboard display
model=wideresnet_quan
#model=vgg11_bn_quan
#model=resnet20_quan
dataset=cifar10
epochs=160
label_info=binarized
test_batch_size=128

attack_sample_size=128 # number of data used for BFA
n_iter=100 # number of iteration to perform BFA
k_top=100 # only check k_top weights with top gradient ranking in each layer

model_path=./save/${DATE}/${dataset}_${model}_${epochs}_${label_info}
save_path=${model_path}/BFA_defense_test
tb_path=${save_path}/tb_log  #tensorboard log path

# set the pretrained model path
#pretrained_model=/home/elliot/Documents/CVPR_2020/BFA_defense_camera_ready/BFA/save/2020-05-08/cifar10_resnet20_quan_160_SGD_binarized/model_best.pth.tar

# resnet20_bin
pretrained_model=${model_path}/state_dict_only.pth
#pretrained_model=${model_path}/model_best.pth.tar

############### Neural network ############################
COUNTER=0
{
while [ $COUNTER -lt 1 ]; do
    $PYTHON main36.py --dataset ${dataset} \
        --data_path ${data_path}   \
        --arch ${model} --save_path ${save_path}  \
        --test_batch_size ${test_batch_size} --workers 4 \
        --print_freq 50 \
        --evaluate --resume ${pretrained_model} \
        --reset_weight --bfa --n_iter ${n_iter} \
        --challenge_batch_size ${attack_sample_size} \

    let COUNTER=COUNTER+1
done
} 
