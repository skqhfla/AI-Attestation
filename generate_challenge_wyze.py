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

def update_image_to_wyze_uniform(model, images, num_classes=5, max_iters=1000, step_size=0.03):
    """특정 그리드 셀을 고정하고 분류 무결성만 정밀 타격하는 '고정 그리드 챌린지' 생성"""
    T = torch.full((num_classes,), 1.0 / num_classes, device=DEVICE)
    
    obj_idx = [4, 14, 24]
    class_idx = [5,6,7,8,9, 15,16,17,18,19, 25,26,27,28,29]
    
    # [Target Locking] 중앙 지점 인근의 고정된 셀을 검증 구역으로 지정 (Head 16의 중앙 근처)
    # Head 32(336개) + Head 16(1344개) 중 총 1008번 인덱스 주변을 사용
    FIXED_TARGET_IDX = 1008 
    
    for i in range(max_iters):
        images.requires_grad = True
        d32, d16 = model(images)
        
        all_probs = []
        all_objs = [] # 모니터링용
        
        for head in [d32, d16]:
            all_objs.append(sigmoid(head[obj_idx]).reshape(-1))
            cls = sigmoid(head[class_idx].view(3, 5, head.shape[1], head.shape[2]))
            all_probs.append(cls.permute(0, 2, 3, 1).reshape(-1, 5))
            
        objs = torch.cat(all_objs)
        probs = torch.cat(all_probs)
        
        # 2. 고정 지점의 분류 확률 벡터 추출
        P = probs[FIXED_TARGET_IDX]
        
        # (A) L_KL: KL Divergence
        plus_eps = 1e-10
        loss_kl = (P * torch.log((P + plus_eps) / (T + plus_eps))).sum()
        
        # (B) L_L2: L2 Distance
        loss_l2 = torch.norm(P - T, p=2)
        
        # (C) L_unif: -Variance
        loss_unif = -torch.var(P)
        
        # [Strategy] 탐지 임계값(Obj)은 무시하고 분류 레이어의 그라디언트에만 집중
        # 단, Objectness가 너무 낮아지지 않도록 최소한의 유지 항 사용 (옵션)
        loss_obj = (objs[FIXED_TARGET_IDX] - 0.5) ** 2 
        
        # 최종 손실 조합 (논문 가중치 준수)
        loss = loss_kl + 0.1 * loss_l2 + 0.1 * loss_unif + 0.1 * loss_obj
        
        if (i+1) % 200 == 0:
            status = ", ".join([f"{p.item()*100:4.1f}%" for p in P])
            print(f"  [Iter {i+1:04d}] Loss: {loss.item():.6f} | Obj @ Fixed: {objs[FIXED_TARGET_IDX].item():.3f} | P: [{status}]")

        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            images -= step_size * images.grad.sign()
            images = torch.clamp(images, 0, 1)
            
    return images.detach()

def load_and_predict_wyze(model, save_dir):
    """탐지 성공 여부와 상관없이 고정된 검증 그리드의 클래스 확률을 직접 리포팅"""
    transform = transforms.Compose([transforms.ToTensor()])
    image_paths = sorted([os.path.join(save_dir, img) for img in os.listdir(save_dir) if img.endswith('.png')])
    
    if not image_paths:
        print("No challenge images found.")
        return

    # 최적화 시 사용했던 고정 지점과 일치해야 함
    FIXED_TARGET_IDX = 1008
    print(f"\nEvaluating Fixed-Grid Classification (Index: {FIXED_TARGET_IDX})...")
    
    model.eval()
    obj_idx = [4, 14, 24]
    class_idx = [5,6,7,8,9, 15,16,17,18,19, 25,26,27,28,29]
    
    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            d32, d16 = model(img_tensor)
            
            all_objs = []
            all_probs = []
            for head in [d32, d16]:
                all_objs.append(sigmoid(head[obj_idx]).reshape(-1))
                p = sigmoid(head[class_idx].view(3, 5, head.shape[1], head.shape[2]))
                all_probs.append(p.permute(0, 2, 3, 1).reshape(-1, 5))
            
            objs = torch.cat(all_objs)
            probs = torch.cat(all_probs)
            
            # 고정된 검증 지점의 실제 클래스 확률 직접 추출
            target_probs = probs[FIXED_TARGET_IDX]
            target_obj = objs[FIXED_TARGET_IDX]
            
            prob_str = ", ".join([f"C{j}:{p*100:4.1f}%" for j, p in enumerate(target_probs)])
            print(f"[{i+1:03d}] {os.path.basename(img_path)} | Internal-Obj:{target_obj.item():.4f} | {prob_str}")

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
