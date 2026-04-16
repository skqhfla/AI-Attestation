#!/usr/bin/env sh

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

# Automatic check the host and configuration
case $HOST in
"ubuntu-MS-7B23")
    PYTHON="/home/ubuntu/anaconda3/envs/BFA/bin/python3" # python environment path
    TENSORBOARD='/home/ubuntu/anaconda3/envs/BFA/bin/tensorboard' # tensorboard environment path
    data_path='/home/ubuntu/BFA/BFA/data/imagenet' # dataset path
    ;;
esac

DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save/${DATE}/
fi

############### Configurations ########################
enable_tb_display=false # enable tensorboard display
model=resnet18
dataset=imagenet
test_batch_size=256

save_path=./save/${DATE}/${dataset}_${model}_eval/

tb_path=${save_path}/tb_log  #tensorboard log path

############### Neural network ############################
{
$PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ${save_path} \
    --test_batch_size ${test_batch_size} \
    --workers 8 --ngpu 1 --gpu_id 0 \
    --reset_weight \
    --evaluate
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
