#!/usr/bin/env python3
import os
import random
import time
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import copy
from pathlib import Path

# AI-Attestation 패키지 경로 추가
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.quan_wyze import wyze_resnet20_quan
from attack_wyze.BFA import BFA
from utils import AverageMeter, time_string

# YOLO 전역 설정
CONF_THRESHOLD = 0.25

parser = argparse.ArgumentParser(description="BFA Attack on Wyze YOLO (Unified Flow)")

# main.py 호환 인자
parser.add_argument("--save_path", type=str, default="./results/wyze")
parser.add_argument("--n_iter", type=int, default=20, help="Number of iterations for BFA")
parser.add_argument("--k_top", type=int, default=10, help="Number of bit candidates to search")
parser.add_argument("--manualSeed", type=int, default=None)
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--ngpu", type=int, default=1)
parser.add_argument("--data_path", type=str, default="../dumps_raw", help="Path to dumps_raw")

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

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
log = open(os.path.join(args.save_path, f"log_wyze_seed_{args.manualSeed}.txt"), "w")

def print_log(msg, log):
    print(msg)
    log.write(f"{msg}\n")
    log.flush()

def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-torch.clamp(x, -50, 50)))

# ------------------- Utils -----------------------
def calculate_accuracy(model, data_loader, device):
    """
    Wyze 전용 성능 측정: Success Rate (탐지율) 및 Avg Confidence
    main.py의 validate와 역할을 맞춤.
    """
    model.eval()
    detected_count = 0
    total_count = 0
    all_confs = []
    
    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device)
            d32, d16 = model(inputs)
            
            frame_detected = False
            for head in [d32, d16]:
                # 각 앵커별 클래스 채널 (5-9, 15-19, 25-29)
                class_idx = [5,6,7,8,9, 15,16,17,18,19, 25,26,27,28,29]
                class_probs = torch.stack([sigmoid(head[c]) for c in class_idx])
                
                if (class_probs > CONF_THRESHOLD).any():
                    frame_detected = True
                    # 기준치 이상의 클래스 확신도 평균 기록
                    all_confs.append(class_probs[class_probs > CONF_THRESHOLD].mean().item())
            
            if frame_detected:
                detected_count += 1
            total_count += 1
            
    success_rate = (detected_count / total_count) * 100
    avg_conf = np.mean(all_confs) if all_confs else 0.0
    return success_rate, avg_conf

def validate(data_loader, model, criterion, log, device):
    """main.py의 validate 함수 형식을 따름"""
    success_rate, avg_conf = calculate_accuracy(model, data_loader, device)
    
    # YOLO의 특성상 Top-1 대신 Success Rate를 Prec@1 위치에 표시
    print_log(f"**Test** Success Rate: {success_rate:.2f}% | Avg Conf: {avg_conf:.4f}", log)
    
    # main.py의 output_summary (Top-K) 대신 여기서는 간단한 Metric 반환
    dummy_top10 = [[0]*10] * len(data_loader)
    return success_rate, avg_conf, 0.0, dummy_top10

# ------------------- Attack function -----------------------
def perform_attack(attacker, model, train_loader, val_loader, N_iter, log, save_path, device):
    model.eval()
    attack_time = AverageMeter()
    
    # 1. 공격용 샘플 준비 (기존 dumps_raw 첫 번째 이미지)
    attack_data = train_loader[0].to(device)
    with torch.no_grad():
        d32, d16 = model(attack_data)
        attack_target = (d32.detach().clone(), d16.detach().clone())

    # 2. 초기 평가
    print_log("\n" + "="*40, log)
    print_log("Initial Evaluation before BFA", log)
    initial_acc, initial_conf, _, _ = validate(val_loader, model, attacker.criterion, log, device)
    print_log("="*40, log)

    last_acc = initial_acc
    all_attack_logs = []

    # 3. 공격 루프
    for i_iter in range(N_iter):
        print_log('**********************************', log)
        start_attack = time.time()
        
        # 실제 Bit-Flip 수행
        step_log = attacker.progressive_bit_search(model, attack_data, attack_target)
        
        attack_time.update(time.time() - start_attack)
        
        print_log(
            f"Iteration: [{i_iter+1:03d}/{N_iter:03d}] "
            f"Attack Time {attack_time.val:.3f} ({attack_time.avg:.3f}) "
            f"{time_string()}",
            log
        )
        print_log(f"Loss (Ascending): {attacker.loss_max:.6f}", log)
        print_log(f"Bit Flips (Cumulative): {attacker.bit_counter}", log)

        # 4. 성능 재측정
        current_acc, current_conf, _, _ = validate(val_loader, model, attacker.criterion, log, device)
        acc_drop = last_acc - current_acc
        last_acc = current_acc

        # 5. Attack Profile 데이터 축적 (main.py 형식)
        for row in step_log:
            # row: [module idx, bit-flip idx, module name, weight idx, prior, post]
            # 여기에 acc와 drop 추가 (main.py 컬럼 대응)
            row.append(current_acc)
            row.append(acc_drop)
            row.append(current_conf) # 추가 신뢰도 정보
            all_attack_logs.append(row)

        # 6. 정지 조건 (인식률 10% 이하 등)
        if current_acc <= 10.0 and attacker.bit_counter >= 10:
            print_log(">>> Target criteria met. Ending attack.", log)
            break

    # 7. CSV 저장 (main.py의 attack_profile 형태)
    if all_attack_logs:
        df = pd.DataFrame(all_attack_logs, columns=[
            'module idx', 'bit-flip idx', 'module name', 'weight idx',
            'weight before attack', 'weight after attack',
            'validation accuracy', 'accuracy drop', 'avg_conf'
        ])
        df['trial seed'] = args.manualSeed
        csv_path = os.path.join(save_path, f'attack_profile_{args.manualSeed}.csv')
        df.to_csv(csv_path, index=False)
        print_log(f"Saved attack profile at {csv_path}", log)

def main():
    device = torch.device("cuda" if args.use_cuda else "cpu")
    print_log(f"Using device: {device}", log)

    # 1. 모델 로드
    print_log("Loading Wyze YOLO model...", log)
    model = wyze_resnet20_quan().to(device)
    model.eval()

    # 2. 데이터 로더 준비
    frames_dir = Path(args.data_path)
    if not frames_dir.exists():
        # 다양한 경로 재탐색 (공용 로직 지원)
        for p in [Path("../dumps_raw"), Path("./dumps_raw"), Path("./sample_data")]:
            if p.exists() and list(p.glob("frame_*_input.bin")):
                frames_dir = p
                break

    if not list(frames_dir.glob("frame_*_input.bin")):
        print_log(f"[ERROR] No sample images found at {frames_dir}", log)
        sys.exit(1)

    print_log(f"Loading samples from: {frames_dir}", log)
    input_files = sorted(list(frames_dir.glob("frame_*_input.bin")))[:20]
    
    sample_images = []
    for f in input_files:
        data = np.fromfile(f, dtype=np.int8).reshape(256, 448, 3).transpose(2, 0, 1)
        image_tensor = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
        sample_images.append(image_tensor)
    
    # 3. 손실 함수 (Targeted Class Crushing Loss)
    def yolo_crushing_loss(outputs, targets):
        d32, d16 = outputs
        # 각 앵커의 클래스 채널(5~9, 15~19, 25~29)을 모두 합산
        class_channels = [5,6,7,8,9, 15,16,17,18,19, 25,26,27,28,29]
        loss_val = 0
        for head in [d32, d16]:
            # 시그모이드를 적용하여 실제 확률값으로 변환
            probs = sigmoid(head[class_channels])
            # 확률이 0.1 이상인 지점들만 집중 타격 (배경 노이즈 무시)
            loss_val += probs[probs > 0.1].sum()
        return -loss_val # 확률 합계를 최소화하도록 (음수화)

    # 4. 공격 수행
    attacker = BFA(yolo_crushing_loss, model, args.k_top)
    perform_attack(
        attacker, model, sample_images, sample_images, 
        args.n_iter, log, args.save_path, device
    )

    log.close()

if __name__ == "__main__":
    main()
