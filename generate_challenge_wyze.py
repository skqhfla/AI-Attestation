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

def update_image_to_wyze_uniform(model, images, num_classes=5, max_iters=500, step_size=0.02):
    """PGD를 사용하여 '가장 확실하게 탐지된 단 하나의 객체'의 클래스 확률을 0.2로 최적화"""
    target_prob = 0.2
    obj_idx = [4, 14, 24]
    class_idx = [5,6,7,8,9, 15,16,17,18,19, 25,26,27,28,29]
    
    for i in range(max_iters):
        images.requires_grad = True
        d32, d16 = model(images)
        
        all_obj_scores = []
        all_class_probs = []
        
        for head in [d32, d16]:
            obj = sigmoid(head[obj_idx])
            all_obj_scores.append(obj.reshape(-1))
            cls = sigmoid(head[class_idx].view(3, 5, head.shape[1], head.shape[2]))
            all_class_probs.append(cls.permute(0, 2, 3, 1).reshape(-1, 5))
            
        objs = torch.cat(all_obj_scores)
        probs = torch.cat(all_class_probs)
        
        # 1. 가장 높은 Objectness를 가진 '단 하나의 셀' 선정
        top_val, top_idx = torch.topk(objs, 1)
        target_cell_probs = probs[top_idx[0]] # [5]
        
        # 2. 손실 함수 설계
        # (A) Top-1 셀의 5개 클래스 확률을 정확히 20%(0.2)로 유도
        loss_uniform = ((target_cell_probs - target_prob) ** 2).sum()
        # (B) 해당 셀이 '확실한 물체'로 인식되도록 Objectness 유도 (0.8+)
        loss_obj = (objs[top_idx[0]] - 0.8) ** 2
        # (C) 배경 억제: 나머지 구역의 Objectness를 0으로 강제 (배경 노이즈 제거)
        bg_objs = torch.cat([objs[:top_idx[0]], objs[top_idx[0]+1:]])
        loss_bg = (bg_objs ** 2).mean()
        
        loss = loss_uniform + 0.1 * loss_obj + 0.5 * loss_bg
        
        if (i+1) % 100 == 0:
            status = ", ".join([f"{p.item()*100:4.1f}%" for p in target_cell_probs])
            print(f"  [Iter {i+1:03d}] Loss: {loss.item():.6f} | Obj: {objs[top_idx[0]].item():.3f} | Top-1: [{status}]")

        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            images -= step_size * images.grad.sign()
            images = torch.clamp(images, 0, 1)
            
    return images.detach()

def load_and_predict_wyze(model, save_dir):
    """생성된 챌린지 이미지를 로드하여 '가장 높은 탐지 신뢰도'를 가진 구역의 확률 리포팅"""
    transform = transforms.Compose([transforms.ToTensor()])
    image_paths = sorted([os.path.join(save_dir, img) for img in os.listdir(save_dir) if img.endswith('.png')])
    
    if not image_paths:
        print("No challenge images found.")
        return

    print(f"\nEvaluating Top-1 Detection of {len(image_paths)} challenge images...")
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
            
            # 가장 높은 Objectness를 가진 칸 찾기
            max_val, max_idx = torch.max(objs, 0)
            target_probs = probs[max_idx] # [5]
            
            prob_str = ", ".join([f"C{j}:{p*100:4.1f}%" for j, p in enumerate(target_probs)])
            print(f"[{i+1:03d}] {os.path.basename(img_path)} | Obj:{max_val.item():.3f} | {prob_str}")

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
