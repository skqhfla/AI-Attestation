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

def update_image_to_wyze_uniform(model, images, num_classes=5, max_iters=200, step_size=0.01):
    """PGD를 사용하여 모델의 클래스 출력이 균등 분포(1/N)가 되도록 이미지 최적화"""
    target_dist = torch.full((1, num_classes), 1.0 / num_classes).to(DEVICE)
    class_idx = [5,6,7,8,9, 15,16,17,18,19, 25,26,27,28,29]
    
    for i in range(max_iters):
        images.requires_grad = True
        d32, d16 = model(images)
        
        # 각 헤드에서 클래스 로짓 추출 후 평균화
        all_logits = []
        for head in [d32, d16]:
            # head shape: [30, H, W] -> [15, H, W] -> [3_anchors, 5_classes, H, W]
            c = head[class_idx].view(3, 5, head.shape[1], head.shape[2])
            avg_logits = c.mean(dim=(0, 2, 3)) # 앵커와 그리드 전체에 대해 평균
            all_logits.append(avg_logits)
        
        final_logits = torch.stack(all_logits).mean(dim=0).unsqueeze(0)
        
        # KL Divergence 손실 계산 (Softmax 분포 vs 균등 분포)
        loss = F.kl_div(F.log_softmax(final_logits, dim=1), target_dist, reduction='batchmean')
        
        if (i+1) % 50 == 0:
            print(f"  [Optimization] Iter {i+1}/{max_iters}, Loss: {loss.item():.6f}")

        model.zero_grad()
        loss.backward()
        
        # 이미지 업데이트 (경사 하강 방향)
        with torch.no_grad():
            images -= step_size * images.grad.sign()
            images = torch.clamp(images, 0, 1)
            
    return images.detach()

def load_and_predict_wyze(model, save_dir):
    """생성된 챌린지 이미지를 로드하여 클래스별 평균 확률 리포팅"""
    transform = transforms.Compose([transforms.ToTensor()])
    image_paths = sorted([os.path.join(save_dir, img) for img in os.listdir(save_dir) if img.endswith('.png')])
    
    if not image_paths:
        print("No challenge images found.")
        return

    print(f"\nEvaluating {len(image_paths)} optimized challenge images...")
    model.eval()
    class_idx = [5,6,7,8,9, 15,16,17,18,19, 25,26,27,28,29]
    
    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            d32, d16 = model(img_tensor)
            
            # 클래스별 평균 확률 계산
            class_probs = []
            for head in [d32, d16]:
                # head[class_idx] -> [15, H, W] -> [3, 5, H, W] -> sigmoid -> mean over (0,2,3) -> [5]
                p = sigmoid(head[class_idx].view(3, 5, head.shape[1], head.shape[2]))
                class_probs.append(p.mean(dim=(0, 2, 3)))
            
            final_avg_probs = torch.stack(class_probs).mean(dim=0)
            prob_str = ", ".join([f"C{j}:{p*100:4.1f}%" for j, p in enumerate(final_avg_probs)])
            print(f"[{i+1:03d}] {os.path.basename(img_path)} | {prob_str}")

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. 모델 로드
    print(f"Loading {MODEL_NAME}...")
    model = wyze_resnet20_quan().to(DEVICE)
    model.eval()

    # 2. 저장 경로 설정
    save_dir = os.path.join("./data/challenge", MODEL_NAME, "optimized")
    os.makedirs(save_dir, exist_ok=True)

    # 3. 기존 파일 인덱싱
    existing_files = os.listdir(save_dir)
    pattern = re.compile(r"challenge_image_(\d+)\.png")
    max_index = 0
    for fname in existing_files:
        match = pattern.match(fname)
        if match:
            num = int(match.group(1))
            if num > max_index: max_index = num
    
    existing_count = len([f for f in existing_files if pattern.match(f)])
    remaining_count = max(0, NUM_CHALLENGES - existing_count)

    if remaining_count == 0:
        print(f"이미 {NUM_CHALLENGES}개의 최적화된 챌린지가 존재합니다.")
    else:
        print(f"{remaining_count}개의 최적화된 챌린지 이미지를 생성합니다...")
        start_index = max_index + 1
        for i in range(remaining_count):
            # 초기 랜덤 노이즈 생성
            img = generate_random_image()
            # PGD 최적화 수행
            optimized_img = update_image_to_wyze_uniform(model, img)
            
            filename = f"challenge_image_{start_index + i}.png"
            save_path = os.path.join(save_dir, filename)
            save_image(optimized_img, save_path)
            if (i+1) % 5 == 0:
                print(f"Generated and Optimized {i+1}/{remaining_count} images.")

    # 4. 결과 로드 및 리포팅
    load_and_predict_wyze(model, save_dir)
    print("\nWyze Challenge generation process finished.")

if __name__ == "__main__":
    main()
