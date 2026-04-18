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

def update_image_to_wyze_uniform(model, images, num_classes=5, max_iters=2000, step_size=0.05):
    """체크보드 패치와 가변 스텝을 사용하여 2,000회 내에 수렴하는 가속 최적화 엔진"""
    # 목표 로짓 설정: Sigmoid(-1.386) ≈ 0.2
    target_cls_logit = -1.386
    target_obj_logit = 1.6 # 약 83% 탐지 목표
    
    obj_idx = [4, 14, 24]
    class_idx = [5,6,7,8,9, 15,16,17,18,19, 25,26,27,28,29]
    FIXED_TARGET_IDX = 1022 

    # [1. 초기화] 고대비 체크보드 시드 (Checkerboard Seed)
    # 256x448 이미지의 중앙(Head 16, Row 8, Col 14) 부근에 엣지 정보 주입
    with torch.no_grad():
        h_center, w_center = 8 * 16, 14 * 16
        for hh in range(h_center-8, h_center+8):
            for ww in range(w_center-8, w_center+8):
                # 4x4 크기의 체크보드 패턴 (0.1과 0.9 교차)
                if ((hh // 4) + (ww // 4)) % 2 == 0:
                    images[:, :, hh, ww] = 0.9
                else:
                    images[:, :, hh, ww] = 0.1

    for i in range(max_iters):
        images.requires_grad = True
        d32, d16 = model(images)
        
        all_cls_logits = []
        all_obj_logits = []
        for head in [d32, d16]:
            all_obj_logits.append(head[obj_idx].reshape(-1))
            cls = head[class_idx].view(3, 5, head.shape[1], head.shape[2])
            all_cls_logits.append(cls.permute(0, 2, 3, 1).reshape(-1, 5))
            
        objs_logit = torch.cat(all_obj_logits)
        probs_logit = torch.cat(all_cls_logits)
        
        target_P_logit = probs_logit[FIXED_TARGET_IDX]
        target_O_logit = objs_logit[FIXED_TARGET_IDX]
        
        # (A) 클래스 독립적 최적화 (20% 목표)
        loss_cls = ((target_P_logit - target_cls_logit) ** 2).sum()
        
        # (B) 탐지 활성화 손실
        loss_obj = (target_O_logit - target_obj_logit) ** 2
        
        # (C) 배경 억제
        bg_logits = torch.cat([objs_logit[:FIXED_TARGET_IDX], objs_logit[FIXED_TARGET_IDX+1:]])
        loss_bg = (torch.clamp(bg_logits + 5.0, min=0) ** 2).mean()
        
        loss = 2.0 * loss_cls + 1.0 * loss_obj + 0.001 * loss_bg
        
        if (i+1) % 400 == 0:
            current_P = sigmoid(target_P_logit)
            current_O = sigmoid(target_O_logit)
            status = ", ".join([f"{p.item()*100:4.1f}%" for p in current_P])
            print(f"  [Iter {i+1:04d}] Loss: {loss.item():.4f} | Obj: {current_O.item():.3f} | P: [{status}]")

        model.zero_grad()
        loss.backward()
        
        # [2. 다이내믹 스텝] Warm-up 전략
        # 초반 300회는 0.1로 강력하게 밀고, 이후 0.05로 정밀 튜닝
        current_step = step_size * 2.0 if i < 300 else step_size
        
        if images.grad is not None:
            with torch.no_grad():
                images -= current_step * images.grad.sign()
                images = torch.clamp(images, 0, 1)
            
    return images.detach()

def load_and_predict_wyze(model, save_dir):
    """최종 생성된 가속 챌린지 지표 리포팅"""
    transform = transforms.Compose([transforms.ToTensor()])
    image_paths = sorted([os.path.join(save_dir, img) for img in os.listdir(save_dir) if img.endswith('.png')])
    
    if not image_paths:
        return

    FIXED_TARGET_IDX = 1022
    print(f"\nEvaluating Final Challenges (Index: {FIXED_TARGET_IDX})...")
    
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
            
            target_probs = probs[FIXED_TARGET_IDX]
            target_obj = objs[FIXED_TARGET_IDX]
            
            prob_str = ", ".join([f"C{j}:{p*100:4.1f}%" for j, p in enumerate(target_probs)])
            print(f"[{i+1:03d}] {os.path.basename(img_path)} | Obj:{target_obj.item():.4f} | {prob_str}")

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
