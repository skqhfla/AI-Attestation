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
    """[최종 하이퍼파라미터 최적화] 방법론 유지 + 양자화 지형 돌파 스케줄링"""
    # 목표 분포: [0.2, 0.2, 0.2, 0.2, 0.2]
    target_dist = torch.full((num_classes,), 1.0 / num_classes, device=DEVICE)
    
    obj_idx = [4, 14, 24]
    class_idx = [5,6,7,8,9, 15,16,17,18,19, 25,26,27,28,29]
    FIXED_TARGET_IDX = 1022 

    # [1. 초기화] 발산 방지를 위해 중간값(0.5) + 미세 가우시안 노이즈로 출발
    with torch.no_grad():
        h_center, w_center = 8 * 16, 14 * 16
        images[:, :, h_center-8:h_center+8, w_center-8:w_center+8] = 0.5 + torch.randn((1, 3, 16, 16), device=DEVICE) * 0.05
        images = torch.clamp(images, 0, 1)

    for i in range(max_iters):
        images.requires_grad = True
        d32, d16 = model(images)
        
        # 로짓 추출 및 Softmax 정규화
        all_cls_logits = []
        for head in [d32, d16]:
            cls = head[class_idx].view(3, 5, head.shape[1], head.shape[2])
            all_cls_logits.append(cls.permute(0, 2, 3, 1).reshape(-1, 5))
        probs_logit = torch.cat(all_cls_logits)
        
        target_P_logit = probs_logit[FIXED_TARGET_IDX]
        target_P_probs = F.softmax(target_P_logit, dim=0)
        
        # [2. 논문 공식 적용 (Selection 3.2.1)]
        loss_kl = F.kl_div(F.log_softmax(target_P_logit, dim=0), target_dist, reduction='batchmean')
        loss_l2 = torch.norm(target_P_probs - target_dist, p=2)
        loss_unif = torch.var(target_P_probs)
        
        # 방법론 가중치 준수: KL(1.0) + L2(0.1) + Var(0.1)
        loss = loss_kl + 0.1 * loss_l2 + 0.1 * loss_unif
        
        if (i+1) % 400 == 0:
            status = ", ".join([f"{p.item()*100:4.1f}%" for p in target_P_probs])
            diff = target_P_probs.max() - target_P_probs.min()
            print(f"  [PGD-Tuned {i+1:04d}] Loss: {loss.item():.4f} | Gap: {diff.item()*100:.1f}% | P: [{status}]")

        model.zero_grad()
        loss.backward()
        
        # [3. Staircase Step Scheduling] 
        # 8비트 양자화 임계값(1/255 ≈ 0.0039) 아래로 떨어지지 않도록 설계
        if i < 800:
            current_step = step_size          # 충분히 큰 스텝으로 계단 탈출 (0.05)
        elif i < 1500:
            current_step = step_size * 0.2    # 정밀 접근 (0.01)
        else:
            current_step = step_size * 0.05   # 최종 안착 (0.0025, 반올림으로 작용)

        if images.grad is not None:
            with torch.no_grad():
                images -= current_step * images.grad.sign()
                images = torch.clamp(images, 0, 1)
            
    return images.detach()

def load_and_predict_wyze(model, save_dir):
    """최종 생성된 이미지를 고정 타겟 지점(1022)에서 리포트"""
    transform = transforms.Compose([transforms.ToTensor()])
    image_paths = sorted([os.path.join(save_dir, img) for img in os.listdir(save_dir) if img.endswith('.png')])
    
    if not image_paths:
        return

    FIXED_TARGET_IDX = 1022
    print(f"\nEvaluating Methodology-Matched Challenges (Index {FIXED_TARGET_IDX})...")
    
    model.eval()
    obj_idx = [4, 14, 24]
    class_idx = [5,6,7,8,9, 15,16,17,18,19, 25,26,27,28,29]
    
    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            d32, d16 = model(img_tensor)
            
            p_logits = []
            for head in [d32, d16]:
                p = head[class_idx].view(3, 5, head.shape[1], head.shape[2])
                p_logits.append(p.permute(0, 2, 3, 1).reshape(-1, 5))
            
            # 최종 리포트에서도 Softmax 기반 확률 분포 출력
            probs = F.softmax(torch.cat(p_logits)[FIXED_TARGET_IDX], dim=0)
            
            prob_str = ", ".join([f"C{j}:{p*100:4.1f}%" for j, p in enumerate(probs)])
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
