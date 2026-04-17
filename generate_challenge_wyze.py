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

def update_image_to_wyze_uniform(model, images, num_classes=5, max_iters=1000, step_size=0.01):
    """로짓 기반 타겟 락킹을 통해 탐지는 보장(Obj > 0.9)하고 클래스는 20% 균등화하도록 최적화"""
    #目标: Sigmoid(4.5) ≈ 0.99 (Objectness), Sigmoid(-1.386) ≈ 0.2 (Class)
    target_obj_logit = 4.5
    target_cls_logit = -1.386
    
    obj_idx = [4, 14, 24]
    class_idx = [5,6,7,8,9, 15,16,17,18,19, 25,26,27,28,29]
    
    target_idx = None # 최적화할 특정 셀(Grid)을 첫 루프에서 고정
    
    for i in range(max_iters):
        images.requires_grad = True
        d32, d16 = model(images)
        
        all_obj_logits = []
        all_class_logits = []
        
        for head in [d32, d16]:
            all_obj_logits.append(head[obj_idx].reshape(-1))
            cls = head[class_idx].view(3, 5, head.shape[1], head.shape[2])
            all_class_logits.append(cls.permute(0, 2, 3, 1).reshape(-1, 5))
            
        objs_logit = torch.cat(all_obj_logits)
        probs_logit = torch.cat(all_class_logits)
        
        # 1. 첫 루프에서 가장 가능성 높은 타겟 셀 고정 (Target Locking)
        if target_idx is None:
            _, target_idx = torch.max(objs_logit, 0)
            target_idx = target_idx.item()
            print(f"  [Target Locking] Index: {target_idx}")

        # 2. 손실 함수 설계 (로짓 영역에서 직접 타겟팅)
        # (A) 타겟 셀의 Objectness를 4.5(약 99%)로 유도
        loss_obj = (objs_logit[target_idx] - target_obj_logit) ** 2
        # (B) 타겟 셀의 5개 클래스 로짓을 -1.386(약 20%)으로 유도
        loss_cls = ((probs_logit[target_idx] - target_cls_logit) ** 2).sum()
        # (C) 배경 억제: 나머지 구역의 Objectness 로짓을 -10.0(거의 0)으로 강제
        # 슬라이싱을 이용해 타겟 외의 모든 로짓 선택
        bg_logits = torch.cat([objs_logit[:target_idx], objs_logit[target_idx+1:]])
        loss_bg = ((bg_logits + 10.0) ** 2).mean() # -10.0 근처로 유도
        
        loss = loss_obj + loss_cls + 0.1 * loss_bg
        
        if (i+1) % 200 == 0:
            current_obj = sigmoid(objs_logit[target_idx]).item()
            current_cls = sigmoid(probs_logit[target_idx])
            status = ", ".join([f"{p.item()*100:4.1f}%" for p in current_cls])
            print(f"  [Iter {i+1:04d}] Loss: {loss.item():.4f} | Obj: {current_obj:.3f} | Class: [{status}]")

        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            images -= step_size * images.grad.sign()
            images = torch.clamp(images, 0, 1)
            
    return images.detach()

def load_and_predict_wyze(model, save_dir):
    """최적화된 챌린지 이미지를 탐지 보장형 관점에서 상세 리포팅"""
    transform = transforms.Compose([transforms.ToTensor()])
    image_paths = sorted([os.path.join(save_dir, img) for img in os.listdir(save_dir) if img.endswith('.png')])
    
    if not image_paths:
        print("No challenge images found.")
        return

    print(f"\nEvaluating Detection-Assumed Classification of {len(image_paths)} images...")
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
            
            # 최대로 감지된 지점 지표 출력
            max_val, max_idx = torch.max(objs, 0)
            target_probs = probs[max_idx]
            
            # 탐지 보장 확인 (Obj > 0.8일 때 OK)
            status = " [OK]" if max_val.item() > 0.8 else "[WARN]"
            prob_str = ", ".join([f"C{j}:{p*100:4.1f}%" for j, p in enumerate(target_probs)])
            print(f"[{i+1:03d}] {os.path.basename(img_path)} | Det:{status} Obj:{max_val.item():.3f} | {prob_str}")

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
