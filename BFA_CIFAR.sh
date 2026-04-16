#!/usr/bin/env bash


############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

# Automatic check the host and configure
case $HOST in
"cpss-desktop")
    PYTHON="/home/cpss/anaconda3/envs/torch25/bin/python3" # python environment path
    #PYTHON="/home/cpss/anaconda3/envs/bfa36/bin/python3" # python environment path
    data_path='./data/cifar' # dataset path
    ;;
esac

DATE="2025-09-09"
#DATE="2025-08-28-310"
#DATE=`date +%Y-%m-%d`


if [ ! -d "$DIRECTORY" ]; then
    mkdir -p ./save/${DATE}/
fi

############### Configurations ########################
#model=wideresnet_quan
#model=resnet20_quan
model=vgg11_bn_quan
dataset=cifar10
epochs=160
label_info=binarized
test_batch_size=256

challenge_batch_size=256 # number of data used for BFA
n_iter=100 # number of iteration to perform BFA
k_top=100 # only check k_top weights with top gradient ranking in each layer

model_path=./save/${DATE}/${dataset}_${model}_${epochs}_${label_info}
save_path=${model_path}/BFA_defense_test
tb_path=${save_path}/tb_log  #tensorboard log path

pretrained_model=${model_path}/model_best.pth.tar

############### Neural network ############################
COUNTER=0
{
while [ $COUNTER -lt 1 ]; do
    $PYTHON BFA.py --dataset ${dataset} \
        --data_path ${data_path}   \
        --arch ${model} --save_path ${save_path}  \
        --test_batch_size ${test_batch_size} --workers 4 \
        --resume ${pretrained_model} \
        --n_iter ${n_iter} \
        --challenge_batch_size ${challenge_batch_size} \

    let COUNTER=COUNTER+1
done
}

 
#        --ngpu 1 --gpu_id 0 \
