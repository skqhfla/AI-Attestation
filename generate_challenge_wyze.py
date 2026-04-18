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

def update_image_to_wyze_uniform(model, images, num_classes=5, max_iters=1000, step_size=0.05):
    """[DACA 엔진] 탐지 성공을 가정하고 분류 헤드의 20% 균등 분포만을 정밀 유도"""
    # 목표 로짓: Sigmoid(-1.386) ≈ 20%
    target_cls_logit = -1.386
    
    obj_idx = [4, 14, 24]
    class_idx = [5,6,7,8,9, 15,16,17,18,19, 25,26,27,28,29]
    FIXED_TARGET_IDX = 1022 # 정중앙 ROI 고정

    # [1. 초기화] 고대비 노이즈 시드 (반응성 확보)
    with torch.no_grad():
        h_center, w_center = 8 * 16, 14 * 16
        noise = (torch.randn((1, 3, 16, 16), device=DEVICE) * 0.4) + 0.5
        images[:, :, h_center-8:h_center+8, w_center-8:w_center+8] = torch.clamp(noise, 0, 1)

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
        P_actual = sigmoid(target_P_logit)
        
        # [2. 순수 분류 손실 함수 (DAC)]
        # 탐지(Obj)는 방해하지 않도록 관찰만 하고, 분류 가중치만 압박
        loss_mse = ((target_P_logit - target_cls_logit) ** 2).sum()
        loss_var = torch.var(P_actual)
        
        # 배경 억제는 최소화 (타겟 보호)
        loss_bg = (torch.clamp(objs_logit + 5.0, min=0) ** 2).mean()
        
        # 균등성(Variance)에 20배 높은 가중치 부여하여 5개 클래스를 수평으로 정렬
        loss = 2.0 * loss_mse + 20.0 * loss_var + 0.001 * loss_bg
        
        if (i+1) % 200 == 0:
            status = ", ".join([f"{p.item()*100:4.1f}%" for p in P_actual])
            diff = P_actual.max() - P_actual.min()
            print(f"  [DAC Opt {i+1:04d}] Loss: {loss.item():.4f} | Var-Gap: {diff.item()*100:.1f}% | P: [{status}]")

        model.zero_grad()
        loss.backward()
        
        # [3. 단계적 학습률 감쇄]
        current_step = step_size if i < 700 else step_size * 0.2
        
        if images.grad is not None:
            with torch.no_grad():
                images -= current_step * images.grad.sign()
                images = torch.clamp(images, 0, 1)
            
    return images.detach()

def load_and_predict_wyze(model, save_dir):
    """지정된 중앙 ROI(1022)의 분류 무결성 지표 최종 리포팅"""
    transform = transforms.Compose([transforms.ToTensor()])
    image_paths = sorted([os.path.join(save_dir, img) for img in os.listdir(save_dir) if img.endswith('.png')])
    
    if not image_paths:
        return

    FIXED_TARGET_IDX = 1022
    print(f"\nEvaluating Classification Integrity (DAC Assumed @ Index {FIXED_TARGET_IDX})...")
    
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
            print(f"[{i+1:03d}] {os.path.basename(img_path)} | (Internal-Obj:{target_obj.item():.4f}) | {prob_str}")

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
