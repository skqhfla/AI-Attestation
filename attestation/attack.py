import os
import time
import copy
import timm
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from PIL import Image
from torchvision.models import vit_b_16
from torchvision.models import efficientnet_v2_l
from torch.utils.tensorboard import SummaryWriter

from libs import BFA
from libs import create_directory
from libs import quan_Conv2d, quan_Linear
from libs import (
    AverageMeter, RecorderMeter, time_string, convert_secs2time, 
    clustering_loss, change_quan_bitwidth
)

# model_names = sorted(name for name in my_models.__dict__ 
#         if name.islower() and not name.startswith("__") and callable(my_models.__dict__[name]))

parser = argparse.ArgumentParser(description='Training network for image classification (CIFAR10/100 only)')

parser.add_argument("--data_path", type=str, default='./data', help='Dataset path')
parser.add_argument("--dataset", type=str, choices=['cifar10', 'cifar100'], required=True)
# parser.add_argument("--arch", type=str, choices=model_names, required=True)
parser.add_argument("--arch", type=str, choices=['efficientnetv2', 'vit', 'tresnet'], required=True)
parser.add_argument("--test_batch_size", type=int, default=16)
parser.add_argument("--print_freq", type=int, default=100)
parser.add_argument("--save_path", type=str, default='./results')
parser.add_argument("--resume", type=str, default='')
parser.add_argument("--ngpu", type=int, default=1)
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--manualSeed", type=int, default=None)
parser.add_argument("--challenge_batch_size", type=int, default=16)
parser.add_argument("--n_iter", type=int, default=100)
parser.add_argument("--k_top", type=int, default=100)

args = parser.parse_args()
image_counter = 0

# ------------------- Setup -----------------------
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# if args.ngpu == 1:
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

cudnn.benchmark = True

# ------------------- Logging -----------------------
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

log = open(os.path.join(args.save_path, f"log_seed_{args.manualSeed}.txt"), "w")


def print_log(msg, log):
    print(msg)
    log.write(f"{msg}\n")
    log.flush()


def get_dataset_transforms(dataset):
    # mean_std = {
    #     "cifar10": ([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]]),
    #     "cifar100": ([x / 255 for x in [129.3, 124.1, 112.4]], [x / 255 for x in [68.2, 65.4, 70.4]])
    # }
    mean_std = {
        "cifar10": (
            [x / 255 for x in [125.3, 123.0, 113.9]],
            [x / 255 for x in [63.0, 62.1, 66.7]]
        ),
        "cifar100": (
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        )
    }

    mean, std = mean_std[dataset]

    if dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif dataset == 'cifar100':
        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError(f'Invalid dataset name: {dataset}')
    
    return train_transform, test_transform


def get_dataloader(dataset_name, data_path, workers, batch_size, attack_sample_size):
    train_transform, test_transform = get_dataset_transforms(dataset_name)
    dataset_cls = dset.CIFAR10 if dataset_name == "cifar10" else dset.CIFAR100
    num_classes = 10 if dataset_name == "cifar10" else 100

    train_data = dataset_cls(data_path, train=True, transform=train_transform, download=True)
    test_data = dataset_cls(data_path, train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=attack_sample_size, shuffle=True,
            num_workers=workers, pin_memory=True
            )
    test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True
            )
    return train_loader, test_loader, num_classes


def validate(val_loader, model, criterion, log):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    output_summary = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            if targets.ndim == 2:
                targets = torch.argmax(targets, dim=1)

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


def accuracy(output, target, topk=(1,)):
    """Computes the top-k accuracy"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)  # top-k prediction
    pred = pred.t()  # transpose
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # compare predictions with labels

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


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

    global image_counter
    # 각 이미지에 대해 상위 10개 클래스와 확률 출력
    for i in range(challenge_data.size(0)):
        image_counter += 1
        print_log(f"Predictions for Challenge Image {image_counter + 1}:", log)
        for j in range(topk):
            class_idx = topk_indices[i, j].item()
            class_prob = topk_probs[i, j].item() * 100
            print_log(f"Top {j + 1} Prediction: Class {class_idx} with confidence {class_prob:.2f}%", log)
        print_log("\n", log)
    return topk_indices.cpu().numpy()


def perform_attack(attacker, model, model_clean, train_loader, test_loader,
                   N_iter, log, writer, csv_save_path=None,
                   random_attack=False, num_classes=None, device=None):
    global image_counter
    model.eval()
    losses = AverageMeter()
    iter_time = AverageMeter()
    attack_time = AverageMeter()
    df = pd.DataFrame()

    # 공격용 샘플 준비
    for _, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.no_grad():
            _, target = model(data).max(1)
        break

    # origin top-k 추출
    val_acc_top1, val_acc_top5, val_loss, origin_output = validate(
        test_loader, model, attacker.criterion, log
    )
    #origin_df = pd.DataFrame(origin_output, columns=[f'origin_top{i+1}' for i in range(10)])

    print_log(f'**Initial Test** Prec@1: {val_acc_top1:.2f} Prec@5: {val_acc_top5:.2f}', log)

    # time 측정 시작
    end = time.time()

    # iteration별 결과를 쌓을 리스트
    all_iterations = []

    # Challenge 이미지 테스트 (선택)
    # challenge_dir = './data/challenge/' + args.arch
    challenge_dir = './challenges/' + args.arch
    transform = transforms.Compose([transforms.ToTensor()])
    challenge_images = []
    image_paths = [os.path.join(challenge_dir, img)
                   for img in os.listdir(challenge_dir) if img.endswith('.png')]
    for img_path in image_paths:
        img = Image.open(img_path)
        img_tensor = transform(img).unsqueeze(0)
        challenge_images.append(img_tensor)
    if challenge_images:
        print("Challenge_images")
        challenge_data = torch.cat(challenge_images, dim=0).to(device)
        origin_topk_indices = load_and_predict_challenge_images(model, challenge_data, num_classes, device, log)

        image_counter = 0

        origin_df = pd.DataFrame(origin_topk_indices, columns=[f'origin_top{i+1}' for i in range(10)])

    challenge_dir_random = './challenges/' + args.arch + "/random"
    transform_random = transforms.Compose([transforms.ToTensor()])
    challenge_random_images = []
    random_image_paths = [os.path.join(challenge_dir_random, img)
                   for img in os.listdir(challenge_dir_random) if img.endswith('.png')]
    for img_path in random_image_paths:
        img = Image.open(img_path)
        img_tensor = transform_random(img).unsqueeze(0)
        challenge_random_images.append(img_tensor)
    if challenge_random_images:
        challenge_random_data = torch.cat(challenge_random_images, dim=0).to(device)
        random_topk_indices = load_and_predict_challenge_images(model, challenge_random_data, num_classes, device, log)

        image_counter = 0    

        random_df = pd.DataFrame(random_topk_indices, columns=[f'random_top{i+1}' for i in range(10)])


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

        flipped_dfs = []
        if challenge_images:
            print_log("Evaluating model predictions after bit flip on challenge images...", log)
            torch.cuda.empty_cache()

            for start_idx in range(0, challenge_data.size(0), args.challenge_batch_size):
                end_idx = min(start_idx + args.challenge_batch_size, challenge_data.size(0))
                batch = challenge_data[start_idx:end_idx]

                flipped_topk_indices = load_and_predict_challenge_images(model, batch, num_classes, device, log)
                batch_df = pd.DataFrame(flipped_topk_indices, columns=[f'flipped_top{i+1}' for i in range(10)])
                flipped_dfs.append(batch_df)

            flipped_df = pd.concat(flipped_dfs, ignore_index=True)
            image_counter = 0


        random_flipped_dfs = []
        if challenge_random_images:
            print_log("Evaluating model predictions after bit flip on random challenge images...", log)
            torch.cuda.empty_cache()

            for start_idx in range(0, challenge_random_data.size(0), args.challenge_batch_size):
                end_idx = min(start_idx + args.challenge_batch_size, challenge_random_data.size(0))
                batch = challenge_random_data[start_idx:end_idx]

                random_flipped_topk_indices = load_and_predict_challenge_images(model, batch, num_classes, device, log)
                batch_df = pd.DataFrame(random_flipped_topk_indices, columns=[f'random_flipped_top{i+1}' for i in range(10)])
                random_flipped_dfs.append(batch_df)

            random_flipped_df = pd.concat(random_flipped_dfs, ignore_index=True)
            image_counter = 0
        # Validation 후 flipped top-k 획득
        val_acc_top1, val_acc_top5, val_loss, flipped_output = validate(
            test_loader, model, attacker.criterion, log
        )

        # origin과 flipped 병합
        merged_df = pd.concat([origin_df, flipped_df, random_df, random_flipped_df], axis=1)
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

        '''
        # Stop 조건
        if args.dataset == 'cifar10':
            break_acc = 11.0
        elif args.dataset == 'cifar100':
            # break_acc = 9.0
            break_acc = 1.0
        else:
            break_acc = 0.0
        if val_acc_top1 <= break_acc:
            break
        '''

        if attacker.bit_counter == 9:
            break
    # ========== CSV 저장 ==========
    if all_iterations:
        final_df = pd.concat(all_iterations, axis=0, ignore_index=True)
        save_path = os.path.join(args.save_path, f'origin_flipped_topk_with_bitflips_{args.arch}.csv')
        final_df.to_csv(save_path, index=False)
        print_log(f"Saved combined CSV at {save_path}", log)

    # ========== attack profile 저장 ==========
    if csv_save_path is not None:
        column_list = [
            'module idx', 'bit-flip idx', 'module name', 'weight idx',
            'weight before attack', 'weight after attack',
            'validation accuracy', 'accuracy drop'
        ]
        df.columns = column_list
        df['trial seed'] = args.manualSeed
        os.makedirs(csv_save_path, exist_ok=True)
        df.to_csv(os.path.join(csv_save_path, f'attack_profile_{args.manualSeed}.csv'), index=False)
        print(f"save attack_profile attack_profile_{args.manualSeed}.csv")

    return


def quantize(model):
    for name, child in model.named_children():
        if isinstance(child, nn.Conv2d):
            quan_layer = quan_Conv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None
            )
            quan_layer.weight.data = child.weight.data.clone()
            if child.bias is not None:
                quan_layer.bias.data = child.bias.data.clone()
            setattr(model, name, quan_layer)
        elif isinstance(child, nn.Linear):
            quan_layer = quan_Linear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None
            )
            quan_layer.weight.data = child.weight.data.clone()
            if child.bias is not None:
                quan_layer.bias.data = child.bias.data.clone()
            setattr(model, name, quan_layer)
        else:
            quantize(child)


def test_model(model, loader, criterion, device):
    from tqdm import tqdm

    test_loss = 0
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for inputs, targets in tqdm(loader, total=len(loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            hard_targets = torch.argmax(targets, dim=1) if targets.ndim == 2 else targets
            correct += predicted.eq(hard_targets).sum().item()

    acc = 100. * correct / total
    loss = test_loss / total
    print(f'[{args.arch}] Loss: {loss}, Acc: {acc}')


# ------------------- Main -----------------------
def main():
    device = torch.device("cuda" if args.use_cuda else "cpu")
    print_log(f"Using device: {device}", log)

    writer = SummaryWriter(os.path.join(args.save_path, "tb_log", f"run_{args.manualSeed}"))

    train_loader, test_loader, num_classes = get_dataloader(
        args.dataset, args.data_path, args.workers, args.test_batch_size, args.challenge_batch_size
    )
            
    # if args.arch.startswith("wideresnet"):
    #     net = my_models.__dict__[args.arch](depth=28, num_classes=num_classes)
    # else:
    #     net = my_models.__dict__[args.arch](num_classes=num_classes)
    if args.arch == 'efficientnetv2':
        net = efficientnet_v2_l()
        net.classifier = nn.Sequential(
            nn.Dropout(p=0.25, inplace=True),
            nn.Linear(net.classifier[-1].in_features, 100),
        )
    elif args.arch == 'vit':
        net = vit_b_16()
        net.heads.head = nn.Linear(net.heads.head.in_features, 100)
    elif args.arch == 'tresnet':
        net = timm.create_model('tresnet_l', pretrained=True, num_classes=100)
    else:
        raise ValueError(f'Invalid model name: {args.arch}')
    
    quantize(net)
    print_log(f"Created model: {args.arch}", log)
    
    if args.use_cuda:
        net = net.cuda()

    # criterion = torch.nn.CrossEntropyLoss().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0003,
        nesterov=True
    )

    # if args.resume and os.path.isfile(args.resume):
    #     checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    #     net.load_state_dict(checkpoint["state_dict"])
    #     if not args.arch == "vgg11_bn_quan":
    #         optimizer.load_state_dict(checkpoint["optimizer"])
    #     args.start_epoch = checkpoint["epoch"]
    #     print_log(f"Resumed from checkpoint: {args.resume}", log)
    # else:
    #     print_log(f"No checkpoint found at: {args.resume}", log)
    
    weights = torch.load(os.path.join(f'./checkpoints/{args.arch}_cifar100_best_weights.pth'),
                         map_location=device,
                         weights_only=True)
    net.load_state_dict(weights)

    test_model(net, test_loader, criterion, device)

    conv2d_cnt = 0
    linear_cnt = 0

    for m in net.modules():
        if isinstance(m, quan_Conv2d):
            conv2d_cnt += 1
        
        if isinstance(m, quan_Linear):
            linear_cnt += 1
    
    print(f'quan_Conv2d: {conv2d_cnt}, quan_Linear: {linear_cnt}')

    for m in net.modules():
        if isinstance(m, (quan_Conv2d, quan_Linear)):
            m.__reset_stepsize__()
            m.__reset_weight__()

    test_model(net, test_loader, criterion, device)

    attacker = BFA(criterion, net, args.k_top)
    perform_attack(
        attacker, net, copy.deepcopy(net), train_loader, test_loader,
        args.n_iter, log, writer, args.save_path, False,
        num_classes, device
    )

    log.close()


if __name__ == "__main__":
    main()
