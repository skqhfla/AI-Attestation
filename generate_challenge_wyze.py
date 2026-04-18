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
    """[최종 정밀판] KL-Divergence와 동적 보상을 사용하여 모든 클래스를 20%에 명중"""
    target_cls_logit = -1.386
    target_dist = torch.full((num_classes,), 1.0/num_classes, device=DEVICE) # [0.2, 0.2, 0.2, 0.2, 0.2]
    
    obj_idx = [4, 14, 24]
    class_idx = [5,6,7,8,9, 15,16,17,18,19, 25,26,27,28,29]
    FIXED_TARGET_IDX = 1022

    # [1. 초기화] 고대비 노이즈 시드
    with torch.no_grad():
        h_center, w_center = 8 * 16, 14 * 16
        noise = (torch.randn((1, 3, 16, 16), device=DEVICE) * 0.5) + 0.5
        images[:, :, h_center-8:h_center+8, w_center-8:w_center+8] = torch.clamp(noise, 0, 1)

    for i in range(max_iters):
        images.requires_grad = True
        d32, d16 = model(images)
        
        # 로짓 추출
        all_cls_logits = []
        for head in [d32, d16]:
            cls = head[class_idx].view(3, 5, head.shape[1], head.shape[2])
            all_cls_logits.append(cls.permute(0, 2, 3, 1).reshape(-1, 5))
        probs_logit = torch.cat(all_cls_logits)
        
        target_P_logit = probs_logit[FIXED_TARGET_IDX]
        P_actual = sigmoid(target_P_logit)
        
        # [2. 복합 정밀 손실 함수]
        # (A) KL-Divergence: 전체적인 분포의 '모양'을 20/20으로 강제
        # YOLO의 독립 출력을 확률 분포로 정규화하여 비교
        P_dist = F.softmax(target_P_logit, dim=0)
        loss_kl = F.kl_div(P_dist.log().unsqueeze(0), target_dist.unsqueeze(0), reduction='batchmean')
        
        # (B) MSE with Class-Boost: 낙오자(10% 미만)에게 10배 가중치 부여
        boost = torch.where(P_actual < 0.10, 10.0, 1.0)
        loss_mse = (boost * (target_P_logit - target_cls_logit) ** 2).sum()
        
        # (C) 전역 균등성 (Variance)
        loss_var = torch.var(P_actual)
        
        # 배경 억제 최소화
        obj_all = torch.cat([h[obj_idx].reshape(-1) for h in [d32, d16]])
        loss_bg = (torch.clamp(obj_all + 5.0, min=0) ** 2).mean()
        
        loss = 50.0 * loss_kl + 2.0 * loss_mse + 20.0 * loss_var + 0.001 * loss_bg
        
        if (i+1) % 200 == 0:
            status = ", ".join([f"{p.item()*100:4.1f}%" for p in P_actual])
            diff = P_actual.max() - P_actual.min()
            print(f"  [KL-Div Opt {i+1:04d}] Loss: {loss.item():.4f} | Gap: {diff.item()*100:.1f}% | P: [{status}]")

        model.zero_grad()
        loss.backward()
        
        # [3. 단계적 학습률 및 리프레시]
        current_step = step_size if i < 700 else step_size * 0.2
        
        # 300회마다 정체 구간 탈출을 위한 픽셀 리프레시
        if (i + 1) % 300 == 0:
            with torch.no_grad():
                images += torch.randn_like(images) * 0.02
        
        if images.grad is not None:
            with torch.no_grad():
                images -= current_step * images.grad.sign()
                images = torch.clamp(images, 0, 1)
            
    return images.detach()

def load_and_predict_wyze(model, save_dir):
    """최종 생성된 DAC 챌린지의 분류 무결성 리포트"""
    transform = transforms.Compose([transforms.ToTensor()])
    image_paths = sorted([os.path.join(save_dir, img) for img in os.listdir(save_dir) if img.endswith('.png')])
    
    if not image_paths:
        return

    FIXED_TARGET_IDX = 1022
    print(f"\nEvaluating Classification Fingerprints (Index {FIXED_TARGET_IDX})...")
    
    model.eval()
    obj_idx = [4, 14, 24]
    class_idx = [5,6,7,8,9, 15,16,17,18,19, 25,26,27,28,29]
    
    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            d32, d16 = model(img_tensor)
            
            p_logits = []
            o_logits = []
            for head in [d32, d16]:
                o_logits.append(head[obj_idx].reshape(-1))
                p = head[class_idx].view(3, 5, head.shape[1], head.shape[2])
                p_logits.append(p.permute(0, 2, 3, 1).reshape(-1, 5))
            
            probs = sigmoid(torch.cat(p_logits))[FIXED_TARGET_IDX]
            obj = sigmoid(torch.cat(o_logits))[FIXED_TARGET_IDX]
            
            prob_str = ", ".join([f"C{j}:{p*100:4.1f}%" for j, p in enumerate(probs)])
            print(f"[{i+1:03d}] {os.path.basename(img_path)} | Obj:{obj.item():.4f} | {prob_str}")

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
