import os
import random
import time
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import copy
import pandas as pd
import numpy as np
from PIL import Image
from models.sam import SAM

from utils import (
        AverageMeter, RecorderMeter, time_string, convert_secs2time, 
        clustering_loss, change_quan_bitwidth
        )
from models.quantization import quan_Conv2d, quan_Linear
import models as my_models
from attack.BFA import BFA

from torch.utils.tensorboard import SummaryWriter

model_names = sorted(name for name in my_models.__dict__ 
        if name.islower() and not name.startswith("__") and callable(my_models.__dict__[name]))

parser = argparse.ArgumentParser(description="Training network for image classification (CIFAR10/100 only)")

parser.add_argument("--data_path", type=str, default="./data", help="Dataset path")
parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], required=True)
parser.add_argument("--arch", type=str, choices=model_names, required=True)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--test_batch_size", type=int, default=256)
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--decay", type=float, default=1e-4)
parser.add_argument("--schedule", type=int, nargs="+", default=[80, 120])
parser.add_argument("--gammas", type=float, nargs="+", default=[0.1, 0.1])
parser.add_argument("--print_freq", type=int, default=100)
parser.add_argument("--save_path", type=str, default="./save/")
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--evaluate", action="store_true")
parser.add_argument("--model_only", action="store_true")
parser.add_argument("--ngpu", type=int, default=1)
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--manualSeed", type=int, default=None)
parser.add_argument("--quan_bitwidth", type=int, default=None)
parser.add_argument("--reset_weight", action="store_true")
parser.add_argument("--enable_bfa", action="store_true")
parser.add_argument("--attack_sample_size", type=int, default=128)
parser.add_argument("--n_iter", type=int, default=20)
parser.add_argument("--k_top", type=int, default=None)
parser.add_argument("--random_bfa", action="store_true")
parser.add_argument("--clustering", action="store_true")
parser.add_argument("--lambda_coeff", type=float, default=1e-3)

args = parser.parse_args()

# ------------------- Setup -----------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.ngpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

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
    mean_std = {
            "cifar10": ([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]]),
            "cifar100": ([x / 255 for x in [129.3, 124.1, 112.4]], [x / 255 for x in [68.2, 65.4, 70.4]])
            }
    mean, std = mean_std[dataset]
    
    if args.arch.startswith('vit'):
    	print("vit transform")
    	train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=28),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])
    	test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])
    else: 
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

# ------------------- Main -----------------------
def main():
    device = torch.device("cuda" if args.use_cuda else "cpu")
    print_log(f"Using device: {device}", log)

    writer = SummaryWriter(os.path.join(args.save_path, "tb_log", f"run_{args.manualSeed}"))

    train_loader, test_loader, num_classes = get_dataloader(
            args.dataset, args.data_path, args.workers, args.test_batch_size, args.attack_sample_size
            )
            
    if args.arch == "effnet_l2":
        net = my_models.__dict__[args.arch](num_classes=num_classes)
    elif args.arch.startswith("wideresnet"):
        net = my_models.__dict__[args.arch](depth=28, num_classes=num_classes)
    else:
        net = my_models.__dict__[args.arch](num_classes=num_classes)

    print_log(f"Created model: {args.arch}", log)
    
    if args.use_cuda:
        net = net.cuda()

    criterion = torch.nn.CrossEntropyLoss().to(device)
    if args.arch == "effnet_l2":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
                net.parameters(),
                base_optimizer,
                lr=args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.decay,
                rho=0.05
                )
    elif args.arch.startswith('vit'):
        print("vit optimizer")
        optimizer = torch.optim.AdamW(
                net.parameters(),
                lr=3e-4,
                weight_decay=0.05,
                )
    else:
        optimizer = torch.optim.SGD(
                net.parameters(),
                lr=args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.decay,
                nesterov=True,
                )

    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(checkpoint["state_dict"])
        if not args.arch == "vgg11_bn_quan":
            optimizer.load_state_dict(checkpoint["optimizer"])
        args.start_epoch = checkpoint["epoch"]
        print_log(f"Resumed from checkpoint: {args.resume}", log)
    else:
        print_log(f"No checkpoint found at: {args.resume}", log)

    if args.quan_bitwidth:
        change_quan_bitwidth(net, args.quan_bitwidth)

    for m in net.modules():
        if isinstance(m, (quan_Conv2d, quan_Linear)):
            m.__reset_stepsize__()
            if args.reset_weight:
                m.__reset_weight__()

    if args.enable_bfa:
        attacker = BFA(criterion, net, args.k_top)
        perform_attack(
                attacker, net, copy.deepcopy(net), train_loader, test_loader,
                args.n_iter, log, writer, args.save_path, args.random_bfa,
                num_classes, device
                )
        return

    if args.evaluate:
        validate(test_loader, net, criterion, log)
        return

    recorder = RecorderMeter(args.epochs)
    start_time = time.time()
    epoch_time = AverageMeter()

    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch)
        print_log(f"\nEpoch [{epoch+1}/{args.epochs}] | LR: {lr:.5f}", log)

        train_acc, train_loss = train(train_loader, net, criterion, optimizer, epoch, device, log)
        val_acc, _, val_loss, _ = validate(test_loader, net, criterion, log)

        recorder.update(epoch, train_loss, train_acc, val_loss, val_acc)
        is_best = val_acc >= recorder.max_accuracy(False)

        checkpoint_state = {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": net.state_dict(),
                "recorder": recorder,
                "optimizer": optimizer.state_dict(),
                }
        save_checkpoint(checkpoint_state, is_best, args.save_path, "checkpoint.pth.tar", log)
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    log.close()


def adjust_learning_rate(optimizer, epoch):
    lr = args.learning_rate
    for gamma, step in zip(args.gammas, args.schedule):
        if epoch >= step:
            lr *= gamma
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def train(train_loader, model, criterion, optimizer, epoch, device, log):
    print(model)
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    is_sam = isinstance(optimizer, SAM)

    end = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if is_sam:
            loss.backward()
            optimizer.first_step(zero_grad=True)
            criterion(model(inputs), targets).backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print_log(
                    f"Epoch [{epoch:03d}][{i:03d}/{len(train_loader)}] "
                    f"Loss {losses.val:.4f} ({losses.avg:.4f}) "
                    f"Prec@1 {top1.val:.2f} ({top1.avg:.2f}) "
                    f"Prec@5 {top5.val:.2f} ({top5.avg:.2f})",
                    log
                    )

    print_log(f"**Train** Prec@1: {top1.avg:.2f} Prec@5: {top5.avg:.2f}", log)
    return top1.avg, losses.avg

def validate(val_loader, model, criterion, log):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    output_summary = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

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

    # 각 이미지에 대해 상위 10개 클래스와 확률 출력
    for i in range(challenge_data.size(0)):
        print_log(f"Predictions for Challenge Image {i + 1}:", log)
        for j in range(topk):
            class_idx = topk_indices[i, j].item()
            class_prob = topk_probs[i, j].item() * 100
            print_log(f"Top {j + 1} Prediction: Class {class_idx} with confidence {class_prob:.2f}%", log)
        print_log("\n", log)


def perform_attack(attacker, model, model_clean, train_loader, test_loader,
                   N_iter, log, writer, csv_save_path=None,
                   random_attack=False, num_classes=None, device=None):

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
            torch.cuda.empty_cache()
            flipped_outputs = model(challenge_data)
            flipped_probabilities = torch.softmax(flipped_outputs, dim=1)
            flipped_top5_probabilities, flipped_top5_indices = torch.topk(flipped_probabilities, 5, dim=1)
            for idx_img, (preds, probs) in enumerate(zip(flipped_top5_indices, flipped_top5_probabilities)):
                print_log(f"Prediction after Bit Flip for Challenge Image {idx_img + 1}: ", log)
                for j in range(5):
                    print_log(
                        f"Top {j+1} Prediction: Class {preds[j].item()} "
                        f"({probs[j].item()*100:.2f}%)", log
                    )

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

def save_checkpoint(state, is_best, save_path, filename, log):
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(save_path, "model_best.pth.tar")
        torch.save(state, best_path)
        print_log(f"=> Saved best model to {best_path}", log)
    else:
        print_log(f"=> Saved checkpoint to {filepath}", log)


if __name__ == "__main__":
    main()

