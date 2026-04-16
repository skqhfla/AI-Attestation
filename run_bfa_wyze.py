#!/usr/bin/env python3
import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

# AI-Attestation 패키지 경로 추가
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.quan_wyze import wyze_resnet20_quan
from attack_wyze.BFA import BFA

# YOLO 설정
CONF_THRESHOLD = 0.25
NUM_CLASSES = 5

def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-torch.clamp(x, -50, 50)))

def calculate_accuracy(model, data_loader, device):
    """
    YOLO 인식을 '정확도'로 환산하는 함수.
    1. 성공률: 객체를 하나라도 탐지한 이미지 비율
    2. 평균 신뢰도: 탐지된 객체들의 평균 Confidence
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
                # obj_score 채널: 4, 14, 24
                obj_scores = torch.stack([sigmoid(head[4]), sigmoid(head[14]), sigmoid(head[24])])
                if (obj_scores > CONF_THRESHOLD).any():
                    frame_detected = True
                    all_confs.append(obj_scores[obj_scores > CONF_THRESHOLD].mean().item())
            
            if frame_detected:
                detected_count += 1
            total_count += 1
            
    success_rate = (detected_count / total_count) * 100
    avg_conf = np.mean(all_confs) if all_confs else 0.0
    return success_rate, avg_conf

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 모델 로드
    print("Loading Wyze YOLO model...")
    model = wyze_resnet20_quan().to(device)
    model.eval()

    # 2. 데이터 준비
    possible_paths = [
        Path("../dumps_raw"),
        Path("./dumps_raw"),
        Path("../../dumps_raw"),
        Path("./sample_data")
    ]
    
    frames_dir = None
    for p in possible_paths:
        if p.exists() and list(p.glob("frame_*_input.bin")):
            frames_dir = p
            break
            
    if frames_dir is None:
        print("\n[ERROR] No sample images found!")
        sys.exit(1)

    print(f"Loading samples from: {frames_dir}")
    input_files = sorted(list(frames_dir.glob("frame_*_input.bin")))[:20]
    
    sample_images = []
    for f in input_files:
        data = np.fromfile(f, dtype=np.int8).reshape(256, 448, 3).transpose(2, 0, 1)
        sample_images.append(torch.from_numpy(data.astype(np.float32)))
    
    attack_data = sample_images[0].unsqueeze(0).to(device)
    val_loader = [img.unsqueeze(0) for img in sample_images]

    with torch.no_grad():
        d32, d16 = model(attack_data)
        attack_target = (d32.detach().clone(), d16.detach().clone())

    # 3. 손실 함수 (신뢰도를 최소화하도록 마이너스 부호 적용)
    def yolo_crushing_loss(outputs, targets):
        d32, d16 = outputs
        # Objectness 점수의 합에 마이너스를 붙여, 신뢰도가 낮아질수록 Loss가 커지게 설정 (BFA는 Loss 최대화 비트를 찾음)
        loss = -(d32[4].sum() + d32[14].sum() + d32[24].sum() + \
                 d16[4].sum() + d16[14].sum() + d16[24].sum())
        return loss

    # 4. BFA 엔진 초기화
    attacker = BFA(yolo_crushing_loss, model, k_top=10)

    print("\n" + "="*40)
    print("Pre-Attack Evaluation")
    acc, conf = calculate_accuracy(model, val_loader, device)
    print(f"Initial Detection Rate: {acc:.2f}%")
    print(f"Initial Avg Confidence: {conf:.4f}")
    print("="*40)

    print("\nStarting Bit-Flip Attack...")
    
    start_time = time.time()
    bit_flips = 0
    attack_profile = []
    iteration = 0

    while not (bit_flips >= 10 and acc <= 10.0) and iteration < 100:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        
        # PBS 수행
        log = attacker.progressive_bit_search(model, attack_data, attack_target)
        # 성능 측정
        acc, conf = calculate_accuracy(model, val_loader, device)
        
        print(f"Loss (Ascending): {attacker.loss_max:.4f}")
        print(f"Bit Flips (Cumulative): {bit_flips}")
        print(f"Detection Rate: {acc:.2f}% | Avg Conf: {conf:.4f}")
        
        for entry in log:
            attack_profile.append(entry + [acc, conf])

    end_time = time.time()
    print("\n" + "="*40)
    print("Attack Finished!")
    print(f"Total Bit Flips: {bit_flips}")
    print(f"Final Detection Rate: {acc:.2f}%")
    print(f"Total Time: {end_time - start_time:.2f} seconds")
    print("="*40)

    # 5. 결과 저장
    df = pd.DataFrame(attack_profile, columns=[
        'module_idx', 'bit_flip_idx', 'module_name', 'weight_idx', 
        'prior_val', 'post_val', 'current_acc'
    ])
    os.makedirs("./results", exist_ok=True)
    csv_path = "./results/wyze_bfa_profile.csv"
    df.to_csv(csv_path, index=False)
    print(f"Attack profile saved to {csv_path}")

if __name__ == "__main__":
    main()
