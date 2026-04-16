#!/usr/bin/env python3
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

# AI-Attestation 패키지 경로 추가
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.quan_wyze import wyze_resnet20_quan

# 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = 'wyze_resnet20_quan'
IMAGE_SIZE = (256, 448) # Height, Width
NUM_CHALLENGES = 100
CONF_THRESHOLD = 0.25

def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-torch.clamp(x, -50, 50)))

def generate_random_image(batch_size=1, channels=3, height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]):
    """랜덤 노이즈 이미지 생성"""
    if np.random.rand() > 0.5:
        # Uniform Noise
        return torch.rand((batch_size, channels, height, width), device=DEVICE)
    else:
        # Gaussian Noise
        base_image = torch.rand((batch_size, channels, height, width), device=DEVICE)
        gaussian_noise = torch.randn((batch_size, channels, height, width), device=DEVICE) * 0.2
        return torch.clamp(base_image + gaussian_noise, min=0, max=1)

def save_image(tensor, filename):
    """텐서를 PNG 파일로 저장"""
    tensor = tensor.squeeze(0).detach().cpu()
    image = transforms.ToPILImage()(tensor)
    image.save(filename)

def load_and_predict_wyze(model, save_dir):
    """생성된 챌린지 이미지를 로드하여 YOLO 탐지 결과 요약 출력"""
    transform = transforms.Compose([transforms.ToTensor()])
    image_paths = sorted([os.path.join(save_dir, img) for img in os.listdir(save_dir) if img.endswith('.png')])
    
    if not image_paths:
        print("No challenge images found to predict.")
        return

    print(f"\nEvaluating {len(image_paths)} challenge images...")
    model.eval()
    
    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            img = Image.open(img_path).convert('RGB')
            # Wyze 모델은 입력을 int8 범위의 float으로 다루는 경향이 있으므로 
            # 일반적인 ToTensor([0,1]) 후 스케일 조절이 필요할 수 있음.
            # 하지만 여기서는 챌린지 지문용이므로 동일한 전처리를 거치는 것이 중요함.
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            
            # 모델 입력 (0~1 범위를 모델이 기대하는 스케일로 변환할 수도 있지만, 
            # 여기서는 챌린지용 입력 그 자체로 사용)
            d32, d16 = model(img_tensor)
            
            # 탐지 성능 요약 (Objectness 기반)
            all_obj_scores = []
            max_conf = 0.0
            detected = False
            
            for head in [d32, d16]:
                # 클래스 채널 인덱스
                class_idx = [5,6,7,8,9, 15,16,17,18,19, 25,26,27,28,29]
                class_scores = torch.stack([sigmoid(head[c]) for c in class_idx])
                
                # 해당 이미지에서 가장 높게 예측된 클래스 확률 추적
                if class_scores.max().item() > max_conf:
                    max_conf = class_scores.max().item()
                
                # 기준치 이상의 클래스 확신이 있는지 확인
                if (class_scores > CONF_THRESHOLD).any():
                    detected = True
            
            print(f"[{i+1:03d}] {os.path.basename(img_path)}: Max Class Conf: {max_conf:.4f} | Classifiable: {detected}")

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. 모델 로드
    print(f"Loading {MODEL_NAME}...")
    model = wyze_resnet20_quan().to(DEVICE)
    model.eval()

    # 2. 저장 경로 설정
    save_dir = os.path.join("./data/challenge", MODEL_NAME, "random")
    os.makedirs(save_dir, exist_ok=True)

    # 3. 기존 파일 인덱싱
    existing_files = os.listdir(save_dir)
    pattern = re.compile(r"challenge_image_(\d+)\.png")
    max_index = 0
    for fname in existing_files:
        match = pattern.match(fname)
        if match:
            num = int(match.group(1))
            if num > max_index:
                max_index = num
    
    existing_count = len([f for f in existing_files if pattern.match(f)])
    remaining_count = max(0, NUM_CHALLENGES - existing_count)

    if remaining_count == 0:
        print(f"이미 {NUM_CHALLENGES}개의 챌린지 이미지가 존재합니다.")
    else:
        print(f"{remaining_count}개의 챌린지 이미지를 생성합니다...")
        start_index = max_index + 1
        for i in range(remaining_count):
            random_image = generate_random_image()
            filename = f"challenge_image_{start_index + i}.png"
            save_path = os.path.join(save_dir, filename)
            save_image(random_image, save_path)
            if (i+1) % 10 == 0:
                print(f"Generated {i+1}/{remaining_count} images.")

    # 4. 결과 로드 및 리포팅
    load_and_predict_wyze(model, save_dir)
    print("\nWyze Challenge generation process finished.")

if __name__ == "__main__":
    main()
