from __future__ import division
from __future__ import absolute_import

import os, sys, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, clustering_loss, change_quan_bitwidth
from torch.utils.tensorboard import SummaryWriter

import models as my_models
from models.quantization import quan_Conv2d, quan_Linear, quantize

#from models.vanilla_models.vanilla_resnet_imagenet import resnet18
#from models.quan_resnet_imagenet import resnet18_quan
#from models.quan_resnet_imagenet import *

#import torchvision.models as torch_models

from attack36.BFA import *
import torch.nn.functional as F
import copy

import pandas as pd
import numpy as np

from PIL import Image

model_names = sorted(name for name in my_models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(my_models.__dict__[name]))


print("@@@@@@@ Available models", model_names)

################# Options ##################################################
############################################################################
parser = argparse.ArgumentParser(
    description='Training network for image classification',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_path',
                    default='/home/ubuntu/BFA/BFA/data/cifar',
                    type=str,
                    help='Path to dataset')
parser.add_argument(
    '--dataset',
    type=str,
    choices=['cifar10', 'cifar100'],
    help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch',
                    metavar='ARCH',
                    default='lbcnn',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs',
                    type=int,
                    default=200,
                    help='Number of epochs to train.')
parser.add_argument('--optimizer',
                    type=str,
                    default='SGD',
                    choices=['SGD', 'Adam', 'YF'])
parser.add_argument('--test_batch_size',
                    type=int,
                    default=256,
                    help='Batch size.')
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.001,
                    help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay',
                    type=float,
                    default=1e-4,
                    help='Weight decay (L2 penalty).')
parser.add_argument('--schedule',
                    type=int,
                    nargs='+',
                    default=[80, 120],
                    help='Decrease learning rate at these epochs.')
parser.add_argument(
    '--gammas',
    type=float,
    nargs='+',
    default=[0.1, 0.1],
    help=
    'LR is multiplied by gamma on schedule, number of gammas should be equal to schedule'
)
# Checkpoints
parser.add_argument('--print_freq',
                    default=100,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 200)')
parser.add_argument('--save_path',
                    type=str,
                    default='./save/',
                    help='Folder to save checkpoints and log.')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument(
    '--fine_tune',
    dest='fine_tune',
    action='store_true',
    help='fine tuning from the pre-trained model, force the start epoch be zero'
)
parser.add_argument('--model_only',
                    dest='model_only',
                    action='store_true',
                    help='only save the model without external utils_')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--gpu_id',
                    type=int,
                    default=0,
                    help='device range [0,ngpu-1]')
parser.add_argument('--workers',
                    type=int,
                    default=4,
                    help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
# quantization
parser.add_argument(
    '--quan_bitwidth',
    type=int,
    default=None,
    help='the bitwidth used for quantization')
parser.add_argument(
    '--reset_weight',
    dest='reset_weight',
    action='store_true',
    help='enable the weight replacement with the quantized weight')
# Bit Flip Attack
parser.add_argument('--bfa',
                    dest='enable_bfa',
                    action='store_true',
                    help='enable the bit-flip attack')
parser.add_argument('--challenge_batch_size',
                    type=int,
                    default=128,
                    help='challenge batch size')
parser.add_argument('--n_iter',
                    type=int,
                    default=20,
                    help='number of attack iterations')
parser.add_argument(
    '--k_top',
    type=int,
    default=None,
    help='k weight with top ranking gradient used for bit-level gradient check.'
)
parser.add_argument('--random_bfa',
                    dest='random_bfa',
                    action='store_true',
                    help='perform the bit-flips randomly on weight bits')

# Piecewise clustering
parser.add_argument('--clustering',
                    dest='clustering',
                    action='store_true',
                    help='add the piecewise clustering term.')
parser.add_argument('--lambda_coeff',
                    type=float,
                    default=1e-3,
                    help='lambda coefficient to control the clustering term')


#===========================Pretrained 옵션 추가
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
#==========================================


##########################################################################

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.ngpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        args.gpu_id)  # make only device #gpu_id visible, then

args.use_cuda = False
#args.ngpu > 0 and torch.cuda.is_available()  # check GPU



#============================GPU 확인 코드
# GPU 설정 관련 출력
print(f"Number of GPUs (ngpu): {args.ngpu}")
print(f"Selected GPU ID (gpu_id): {args.gpu_id}")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.ngpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id) # make only device #gpu_id visible

# GPU 사용 가능 여부 확인
args.use_cuda = False
#args.ngpu > 0 and torch.cuda.is_available() # check GPU

print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"Using CUDA: {args.use_cuda}")
#==========================================



# Give a random seed if no manual configuration
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

#if args.use_cuda:
#    torch.cuda.manual_seed_all(args.manualSeed)

#cudnn.benchmark = True


#============================Pretrained Model 사용할 때
'''
# ResNet18 pretrained on ImageNet
resnet18 = my_models.resnet18(pretrained=True)

# VGG16 pretrained on ImageNet
vgg16 = my_models.vgg16(pretrained=True)
'''
#=====================================================


###############################################################################
###############################################################################

device = "cpu"
def main():
    # Init logger6
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(
        os.path.join(args.save_path,
                     'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')),
              log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()),
              log)
    
    # CUDA 사용 가능한지 확인하여 device 설정
    device = torch.device("cpu")   #######~
    print(f"Using device: {device}")                 #########~


    # Init the tensorboard path and writer
    tb_path = os.path.join(args.save_path, 'tb_log',
                           'run_' + str(args.manualSeed))
    
    writer = SummaryWriter(tb_path)

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])

    if args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path,
                                  train=True,
                                  transform=train_transform,
                                  download=True)
        test_data = dset.CIFAR10(args.data_path,
                                 train=False,
                                 transform=test_transform,
                                 download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path,
                                   train=True,
                                   transform=train_transform,
                                   download=True)
        test_data = dset.CIFAR100(args.data_path,
                                  train=False,
                                  transform=test_transform,
                                  download=True)
        num_classes = 100

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.test_batch_size,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=False)

    print_log("=> creating model '{}'".format(args.arch), log)

    # Init model, criterion, and optimizer
    if args.arch.startswith("wideresnet"):
        net = my_models.__dict__[args.arch](depth=28, num_classes=num_classes)
    else:
        net = my_models.__dict__[args.arch](num_classes=num_classes)
    print_log("=> network :\n {}".format(net), log)
    print_log("=> num_classes: {}".format(num_classes), log)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    # separate the parameters thus param groups can be updated by different optimizer
    all_param = [
        param for name, param in net.named_parameters()
        if not 'step_size' in name
    ]

    step_param = [
        param for name, param in net.named_parameters() if 'step_size' in name 
    ]

    if args.optimizer == "SGD":
        print("using SGD as optimizer")
        optimizer = torch.optim.SGD(all_param,
                                    lr=state['learning_rate'],
                                    momentum=state['momentum'],
                                    weight_decay=state['decay'],
                                    nesterov=True)


    #=================================pretrained model 불러오기 추가
    '''
    if args.pretrained:
        print_log(f"=> using pre-trained model '{args.arch}'", log)
        net = my_models.__dict__[args.arch](pretrained=True) 
    else:
        print_log(f"=> creating model '{args.arch}' from scratch", log)
        net = my_models.__dict__[args.arch](num_classes)
    '''

    #=============================================================

    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    recorder = RecorderMeter(args.epochs)  # count number of epoches

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            #checkpoint = torch.load(args.resume)
            state_dict = torch.load(args.resume, map_location='cpu')
       
            net.load_state_dict(state_dict)  #####################################????

            print_log(
                "=> loaded checkpoint '{}'".format(
                    args.resume), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume),
                      log)
    else:
        print_log(
            "=> do not use any checkpoint for {} model".format(args.arch), log)
        
    # Configure the quantization bit-width
    if args.quan_bitwidth is not None:
        change_quan_bitwidth(net, args.quan_bitwidth)

    # update the step_size once the model is loaded. This is used for quantization.
    for m in net.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            # simple step size update based on the pretrained model or weight init
            m.__reset_stepsize__()

    # block for weight reset
    if args.reset_weight:
        for m in net.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                m.__reset_weight__()
                # print(m.weight)

    attacker = BFA(criterion, net, args.k_top)
    net_clean = copy.deepcopy(net)
    # weight_conversion(net)

    if args.enable_bfa:
        print("num_classes =", num_classes) ######################3
        print("device =", device) ######################debugging	
        perform_attack(attacker, net, net_clean, train_loader, test_loader,
                       args.n_iter, log, writer, csv_save_path=args.save_path,
                       random_attack=args.random_bfa, num_classes=num_classes, device=device)
        return

    if args.evaluate:
        _,_,_, output_summary = validate(test_loader, net, criterion, log, summary_output=True)
        pd.DataFrame(output_summary).to_csv(os.path.join(args.save_path, 'output_summary_{}.csv'.format(args.arch)),
                                            header=['top-1 output'], index=False)
        return

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()

    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate, current_momentum = adjust_learning_rate(
            optimizer, epoch, args.gammas, args.schedule)
        # Display simulation time
        need_hour, need_mins, need_secs = convert_secs2time(
            epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(
            need_hour, need_mins, need_secs)

        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f}][M={:1.2f}]'.format(time_string(), epoch, args.epochs,
                                                                                   need_time, current_learning_rate,
                                                                                   current_momentum) \
            + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                               100 - recorder.max_accuracy(False)), log)

        # train for one epoch
        train_acc, train_los = train(train_loader, net, criterion, optimizer,
                                     epoch, log)

        # evaluate on validation set
        val_acc, _, val_los = validate(test_loader, net, criterion, log)
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        is_best = val_acc >= recorder.max_accuracy(False)

        if args.model_only:
            checkpoint_state = {'state_dict': net.state_dict}
        else:
            checkpoint_state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'recorder': recorder,
                'optimizer': optimizer.state_dict(),
            }

        save_checkpoint(checkpoint_state, is_best, args.save_path,
                        'checkpoint.pth.tar', log)

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

        # save addition accuracy log for plotting
        accuracy_logger(base_dir=args.save_path,
                        epoch=epoch,
                        train_accuracy=train_acc,
                        test_accuracy=val_acc)

        # ============ TensorBoard logging ============#

        ## Log the graidents distribution
        for name, param in net.named_parameters():
            name = name.replace('.', '/')
            try:
                writer.add_histogram(name + '/grad',
                                    param.grad.clone().cpu().data.numpy(),
                                    epoch + 1,
                                    bins='tensorflow')
            except:
                pass
            
            try:
                writer.add_histogram(name, param.clone().cpu().data.numpy(),
                                      epoch + 1, bins='tensorflow')
            except:
                pass
            
        total_weight_change = 0 
            
        for name, module in net.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                try:
                    writer.add_histogram(name+'/bin_weight', module.bin_weight.clone().cpu().data.numpy(), epoch + 1,
                                        bins='tensorflow')
                    writer.add_scalar(name + '/bin_weight_change', module.bin_weight_change, epoch+1)
                    total_weight_change += module.bin_weight_change
                    writer.add_scalar(name + '/bin_weight_change_ratio', module.bin_weight_change_ratio, epoch+1)
                except:
                    pass
                
        writer.add_scalar('total_weight_change', total_weight_change, epoch + 1)
        print('total weight changes:', total_weight_change)

        writer.add_scalar('loss/train_loss', train_los, epoch + 1)
        writer.add_scalar('loss/test_loss', val_los, epoch + 1)
        writer.add_scalar('accuracy/train_accuracy', train_acc, epoch + 1)
        writer.add_scalar('accuracy/test_accuracy', val_acc, epoch + 1)
    # ============ TensorBoard logging ============#

    log.close()

# 일반 데이터셋에서 샘플 이미지 10개 추출
def sample_images_from_dataset(data_loader, num_samples, device):
    images, targets = [], []
    for i, (input, target) in enumerate(data_loader):
        if len(images) >= num_samples:
            break
        images.append(input.to(device))
        targets.append(target.to(device))
    return torch.cat(images, dim=0)[:num_samples], torch.cat(targets, dim=0)[:num_samples]


# 샘플 이미지에 대한 예측 결과를 출력하는 함수
def print_predictions(model, images, device, log, title=""):
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
        probs = F.softmax(outputs, dim=1)
        top5_probs, top5_indices = torch.topk(probs, 5, dim=1)

        print_log(f"{title} Predictions:", log)
        for i in range(images.size(0)):
            print_log(f"Sample {i + 1}:", log)
            for j in range(5):
                class_idx = top5_indices[i, j].item()
                confidence = top5_probs[i, j].item() * 100
                print_log(f"Top {j + 1}: Class {class_idx} with {confidence:.2f}% confidence", log)
            print_log("\n", log)


# 생성된 챌린지 이미지를 다시 불러와 모델에 넣고 예측 결과 출력 (Top 10 클래스)
def load_and_predict_challenge_images(model, challenge_data, num_classes, device, log):
    
    print(f"~~~~Using device: {device}")
    # 모델이 GPU에 있다면, 입력 데이터를 GPU로 이동
    challenge_data = challenge_data.to(device)
    model = model.to(device)
    model.eval()
    
    # 입력 데이터와 모델의 가중치 데이터 타입을 일치시킴 
    challenge_data = challenge_data.type(next(model.parameters()).dtype).to(device)
    

    # 모델에 입력하여 예측
    with torch.no_grad():
        outputs = model(challenge_data)
        probs = F.softmax(outputs, dim=1)

    # Top 10 예측 결과 출력
    topk = 10
    _, topk_indices = torch.topk(probs, topk, dim=1)
    topk_probs = probs.gather(1, topk_indices)

    # 각 이미지에 대해 상위 10개 클래스와 확률 출력
    for i in range(challenge_data.size(0)):
        print_log(f"Predictions for Challenge Image {i + 1}:", log)
        for j in range(topk):
            class_idx = topk_indices[i, j].item()
            class_prob = topk_probs[i, j].item() * 100
            print_log(f"Top {j + 1} Prediction: Class {class_idx} with confidence {class_prob:.2f}%", log)
        print_log("\n", log)


### num_classes 추가, device 추가, challenge 예측하는 부분 추가

def perform_attack(attacker, model, model_clean, train_loader, test_loader,
                   N_iter, log, writer, csv_save_path=None, random_attack=False, num_classes=None, device=None):
    # Note that, attack has to be done in evaluation model due to batch-norm.
    # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
  
 
    model.eval()
    losses = AverageMeter()
    iter_time = AverageMeter()
    attack_time = AverageMeter()
    df = pd.DataFrame()


    # attempt to use the training data to conduct BFA
    for _, (data, target) in enumerate(train_loader):
        if args.use_cuda:
            target = target.cuda(async=True)
            data = data.cuda()
        # Override the target to erevent label leaking
        _, target = model(data).data.max(1)
        break



    # evaluate the test accuracy of clean model
    val_acc_top1, val_acc_top5, val_loss, origin_output = validate(test_loader, model,
                                                    attacker.criterion, log)

	
    origin_df = pd.DataFrame(origin_output, columns=[f'origin_top{i+1}' for i in range(10)])
    print_log(f'**Initial Test** Prec@1: {val_acc_top1:.2f} Prec@5: {val_acc_top5:.2f}', log)

    # time 측정 시작
    end = time.time()

    # iteration별 결과를 쌓을 리스트
    all_iterations = []

    # Challenge 이미지 테스트 (선택)
    challenge_dir = './data/challenge/' + args.arch
    transform = transforms.Compose([transforms.ToTensor()])
    challenge_images = []
    image_paths = [os.path.join(challenge_dir, img)
                   for img in os.listdir(challenge_dir) if img.endswith('.png')]
    for img_path in image_paths:
        img = Image.open(img_path)
        img_tensor = transform(img).unsqueeze(0)
        challenge_images.append(img_tensor)
    if challenge_images:
        challenge_data = torch.cat(challenge_images, dim=0).to(device)
        load_and_predict_challenge_images(model, challenge_data, num_classes, device, log)

    last_val_acc_top1 = val_acc_top1

    # ========== Bit Flip Attack 루프 ==========
    for i_iter in range(N_iter):
        print_log('**********************************', log)
        start_attack = time.time()

        # 실제 bit flip 수행
        if not random_attack:
            attack_log = attacker.progressive_bit_search(model, data, target)
        else:
            attack_log = attacker.random_flip_one_bit(model)

        # attack 시간 측정
        attack_time.update(time.time() - start_attack)
        end = time.time()

        # loss 업데이트
        if hasattr(attacker, "loss_max"):
            losses.update(attacker.loss_max, data.size(0))

        print_log(
            f"Iteration: [{i_iter+1:03d}/{N_iter:03d}] "
            f"Attack Time {attack_time.val:.3f} ({attack_time.avg:.3f}) "
            f"{time_string()}",
            log
        )
        try:
            print_log(f'loss before attack: {attacker.loss.item():.4f}', log)
            print_log(f'loss after attack: {attacker.loss_max:.4f}', log)
        except:
            pass

        print_log(f'bit flips (cumulative): {attacker.bit_counter}', log)

        writer.add_scalar('attack/bit_flip', attacker.bit_counter, i_iter + 1)
        writer.add_scalar('attack/sample_loss', losses.avg, i_iter + 1)

        # Challenge 이미지 평가 (선택)
        if challenge_images:
            print_log("Evaluating model predictions after bit flip on challenge images...", log)

            for start_idx in range(0, challenge_data.size(0), args.challenge_batch_size):
                end_idx = min(start_idx + args.challenge_batch_size, challenge_data.size(0))
                batch = challenge_data[start_idx:end_idx]
                
                with torch.no_grad():
                    flipped_outputs = model(batch)
                    flipped_probabilities = torch.softmax(flipped_outputs, dim=1)

                flipped_top5_probabilities, flipped_top5_indices = torch.topk(flipped_probabilities, 5, dim=1)
                for i in range(batch.size(0)):
                    global_idx = start_idx + i
                    print_log(f"Prediction after Bit Flip for Challenge Image {global_idx + 1}: ", log)
                    for j in range(5):
                        cls = flipped_top5_indices[i, j].item()
                        prob = flipped_top5_probabilities[i, j].item() * 100
                        print_log(f"Top {j+1} Prediction: Class {cls} ({prob:.2f}%)", log)

        # Validation 후 flipped top-k 획득
        val_acc_top1, val_acc_top5, val_loss, flipped_output = validate(
            test_loader, model, attacker.criterion, log
        )
        flipped_df = pd.DataFrame(flipped_output, columns=[f'flipped_top{i+1}' for i in range(10)])

        # origin과 flipped 병합
        merged_df = pd.concat([origin_df, flipped_df], axis=1)
        merged_df['iteration'] = i_iter + 1
        merged_df['bit_flips'] = attacker.bit_counter
        merged_df['attack_time_sec'] = attack_time.val
        merged_df['attack_time_avg'] = attack_time.avg

        all_iterations.append(merged_df)

        # acc_drop
        acc_drop = last_val_acc_top1 - val_acc_top1 if last_val_acc_top1 is not None else None
        last_val_acc_top1 = val_acc_top1

        # attack_log에도 acc 기록 (기존 코드 유지)
        for row in attack_log:
            row.append(val_acc_top1)
            row.append(acc_drop)

        df = pd.concat([df, pd.DataFrame(attack_log)], ignore_index=True)

        # iteration 시간 측정
        iter_time.update(time.time() - end)
        print_log(f"iteration Time {iter_time.val:.3f} ({iter_time.avg:.3f})", log)
        end = time.time()

        writer.add_scalar('attack/val_top1_acc', val_acc_top1, i_iter + 1)
        writer.add_scalar('attack/val_top5_acc', val_acc_top5, i_iter + 1)
        writer.add_scalar('attack/val_loss', val_loss, i_iter + 1)

        # Stop 조건
        if args.dataset == 'cifar10':
            break_acc = 11.0
        elif args.dataset == 'cifar100':
            break_acc = 9.0
        else:
            break_acc = 0.0
        if val_acc_top1 <= break_acc:
            break

    # ========== CSV 저장 ==========
    if all_iterations:
        final_df = pd.concat(all_iterations, axis=0, ignore_index=True)
        save_path = os.path.join(args.save_path, f'origin_flipped_topk_with_bitflips_{args.arch}.csv')
        final_df.to_csv(save_path, index=False)
        print_log(f"Saved combined CSV at {save_path}", log)

    # ========== attack profile 저장 ==========
    # (기존 df로부터 별도로 저장하는 로직은 필요시 유지)
    if csv_save_path is not None:
        if not df.empty:
            column_list = [
                'module idx', 'bit-flip idx', 'module name', 'weight idx',
                'weight before attack', 'weight after attack',
                'validation accuracy', 'accuracy drop'
            ]
            df.columns = column_list
            df['trial seed'] = args.manualSeed
            os.makedirs(csv_save_path, exist_ok=True)
            df.to_csv(os.path.join(csv_save_path, f'attack_profile_{args.manualSeed}.csv'), index=False)

    return


# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    ###########~~~~~~~~~~~~~~~~~~~~~ 제대로 데이터셋 로가드되나 확인하는 부분 추
    num_classes = len(train_loader.dataset.classes)
    print(f"Number of classes: {num_classes}")  # 1000이어야 함

    # 샘플 배치 로드하여 확인
    #data_iter = iter(train_loader)
    #images, labels = next(data_iter)

    #print(f"Sample batch shape: {images.shape}")
    #print(f"Sample labels: {labels}")


    ###########~~~~~~~~~~~~~~~~~~~~~ 

    
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device)
        target = target.to(device)
        output = model(input)

        loss = criterion(output, target)
        if args.clustering:
            loss += clustering_loss(model, args.lambda_coeff)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log(
                '  Epoch: [{:03d}][{:03d}/{:03d}]   '
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5) + time_string(), log)
    print_log(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
        .format(top1=top1, top5=top5, error1=100 - top1.avg), log)
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, log):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    output_summary = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            #inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            _, top10_indices = torch.topk(outputs, k=10, dim=1)
            output_summary.extend(top10_indices.cpu().numpy().tolist())

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

    print_log(f"**Test** Prec@1: {top1.avg:.2f} Prec@5: {top5.avg:.2f}", log)
    return top1.avg, top5.avg, losses.avg, output_summary


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename, log):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:  # copy the checkpoint to the best model if it is the best_accuracy
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)
        print_log("=> Obtain best accuracy, and update the best model", log)


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    mu = args.momentum

    if args.optimizer != "YF":
        assert len(gammas) == len(
            schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif args.optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_logger(base_dir, epoch, train_accuracy, test_accuracy):
    file_name = 'accuracy.txt'
    file_path = "%s/%s" % (base_dir, file_name)
    # create and format the log file if it does not exists
    if not os.path.exists(file_path):
        create_log = open(file_path, 'w')
        create_log.write('epochs train test\n')
        create_log.close()

    recorder = {}
    recorder['epoch'] = epoch
    recorder['train'] = train_accuracy
    recorder['test'] = test_accuracy
    # append the epoch index, train accuracy and test accuracy:
    with open(file_path, 'a') as accuracy_log:
        accuracy_log.write(
            '{epoch}       {train}    {test}\n'.format(**recorder))


if __name__ == '__main__':
    main()
