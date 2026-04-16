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
    YOLO 인식을 '정확도'로 환산하는 간단한 함수.
    탐지된 객체가 하나라도 있는 이미지의 비율을 측정합니다.
    공격이 성공하면 객체 탐지율이 급격히 떨어집니다.
    """
    model.eval()
    detected_count = 0
    total_count = 0
    
    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device)
            d32, d16 = model(inputs)
            
            # 배치가 1이라고 가정 (단순화)
            for head in [d32, d16]:
                # head shape: [30, H, W]
                # 30개 채널 중 obj_score는 4, 14, 24번 채널 (앵커 0, 1, 2)
                obj_scores = torch.stack([sigmoid(head[4]), sigmoid(head[14]), sigmoid(head[24])])
                if (obj_scores > CONF_THRESHOLD).any():
                    detected_count += 1
                    break # 이 프레임은 이미 탐지됨
            
            total_count += 1
            
    return (detected_count / total_count) * 100

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 모델 로드
    print("Loading Wyze YOLO model...")
    model = wyze_resnet20_quan().to(device)
    model.eval()

    # 2. 데이터 준비 (dumps_raw 폴더의 프레임 사용)
    # 640x360 RGB 바이너리를 읽어 전처리 (단순화된 버전)
    frames_dir = Path("../dumps_raw")
    input_files = sorted(list(frames_dir.glob("frame_*_input.bin")))[:20] # 상위 20프레임 사용
    
    sample_images = []
    for f in input_files:
        data = np.fromfile(f, dtype=np.int8).reshape(256, 448, 3).transpose(2, 0, 1)
        sample_images.append(torch.from_numpy(data.astype(np.float32)))
    
    # 공격용 1장 + 검증용 나머지
    attack_data = sample_images[0].unsqueeze(0).to(device)
    val_loader = [img.unsqueeze(0) for img in sample_images]

    # 공격용 타겟 레이블 (현재 출력을 타겟으로 설정하여 'Crushing' 유도)
    with torch.no_grad():
        d32, d16 = model(attack_data)
        attack_target = (d32.detach().clone(), d16.detach().clone())

    # 3. 손실 함수 정의 (Objectness 점수 자체를 낮추도록 설계)
    def yolo_crushing_loss(outputs, targets):
        d32, d16 = outputs
        # 앵커별 objectness 채널 (4, 14, 24) 합계
        loss = d32[4].sum() + d32[14].sum() + d32[24].sum() + \
               d16[4].sum() + d16[14].sum() + d16[24].sum()
        return loss

    # 4. BFA 엔진 초기화
    # k_top=10 설정 (범위 내 상위 10개 그라디언트 비트 검색)
    attacker = BFA(yolo_crushing_loss, model, k_top=10)

    print("\nStarting Bit-Flip Attack...")
    print(f"Target Criteria: Min 10 flips AND Accuracy < 10.0%")
    
    start_time = time.time()
    bit_flips = 0
    accuracy = calculate_accuracy(model, val_loader, device)
    print(f"Initial Detection Rate: {accuracy:.2f}%")

    attack_profile = []
    iteration = 0

    # 공격 루프: 비트 10개 이상 AND 정확도 10% 이하 달성 시 종료
    while not (bit_flips >= 10 and accuracy <= 10.0) and iteration < 100:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        
        # PBS 수행 (가장 영향력 있는 비트 1개 반전)
        log = attacker.progressive_bit_search(model, attack_data, attack_target)
        bit_flips = attacker.bit_counter
        
        # 정확도(탐지율) 재측정
        accuracy = calculate_accuracy(model, val_loader, device)
        
        print(f"Bit Flips (Cumulative): {bit_flips}")
        print(f"Current Detection Rate: {accuracy:.2f}%")
        
        for entry in log:
            attack_profile.append(entry + [accuracy])

    end_time = time.time()
    print("\n" + "="*40)
    print("Attack Finished!")
    print(f"Total Bit Flips: {bit_flips}")
    print(f"Final Detection Rate: {accuracy:.2f}%")
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
