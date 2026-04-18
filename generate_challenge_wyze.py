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

def update_image_to_wyze_uniform(model, images, num_classes=5, max_iters=2000, step_size=0.01):
    """[방법론 고수 + 정밀 수렴판] KL-Divergence를 유지하되 최적화 안정성 극대화"""
    target_dist = torch.full((num_classes,), 1.0 / num_classes, device=DEVICE)
    class_idx = [5,6,7,8,9, 15,16,17,18,19, 25,26,27,28,29]
    FIXED_TARGET_IDX = 1022 
    T = 2.0  # 온도 스케일링: 기울기를 부드럽게 하여 양자화 계단을 극복

    with torch.no_grad():
        h_center, w_center = 8 * 16, 14 * 16
        images[:, :, h_center-8:h_center+8, w_center-8:w_center+8] = 0.5 + torch.randn((1, 3, 16, 16), device=DEVICE) * 0.02
        images = torch.clamp(images, 0, 1)

    momentum = torch.zeros_like(images)
    mu = 0.9 

    for i in range(max_iters):
        images.requires_grad = True
        d32, d16 = model(images)
        
        all_cls_logits = []
        for head in [d32, d16]:
            cls = head[class_idx].view(3, 5, head.shape[1], head.shape[2])
            all_cls_logits.append(cls.permute(0, 2, 3, 1).reshape(-1, 5))
        probs_logit = torch.cat(all_cls_logits)
        
        # 1. 로짓 센터링: 특정 클래스 독주 방지
        target_P_logit = probs_logit[FIXED_TARGET_IDX]
        target_P_logit = target_P_logit - target_P_logit.mean() 
        
        # 2. 온도 스케일링 적용 Softmax (최적화용)
        target_P_probs_T = F.softmax(target_P_logit / T, dim=0)
        target_P_probs_raw = F.softmax(target_P_logit, dim=0) # 모니터링용
        
        # 3. 동적 가중치: 낙오된 클래스(15% 미만)에게 더 높은 학습 우선순위 부여
        # 이는 수식을 바꾸는 것이 아니라, 기울기 전달 강도를 조절하는 최적화 기법임
        compensation = torch.where(target_P_probs_raw < 0.15, 5.0, 1.0)
        
        # [BAIV 공식] KL(1.0) + L2(0.1) + Var(0.1)
        # 온도 스케일링된 분포로 KL 계산 (안정성 확보)
        loss_kl = F.kl_div(F.log_softmax(target_P_logit / T, dim=0), target_dist, reduction='batchmean')
        loss_l2 = torch.norm(target_P_probs_T - target_dist, p=2)
        loss_unif = torch.var(target_P_probs_T)
        
        # 동적 보상을 통한 최종 손실 구성
        loss = (compensation * loss_kl).sum() + 0.1 * loss_l2 + 0.1 * loss_unif
        
        if (i+1) % 400 == 0:
            status = ", ".join([f"{p.item()*100:4.1f}%" for p in target_P_probs_raw])
            diff = target_P_probs_raw.max() - target_P_probs_raw.min()
            print(f"  [BAIV-Stable {i+1:04d}] Gap: {diff.item()*100:.1f}% | P: [{status}]")

        model.zero_grad()
        loss.backward()
        
        if images.grad is not None:
            grad = images.grad
            grad = grad / (torch.mean(torch.abs(grad)) + 1e-10)
            momentum = mu * momentum + grad
            
            with torch.no_grad():
                images -= step_size * momentum.sign() 
                images = torch.clamp(images, 0, 1)
        
        if (i+1) % 500 == 0:
            step_size *= 0.5
            
    return images.detach()

def update_image_to_wyze_multi_label(model, images, num_classes=5, max_iters=1000, step_size=0.01):
    """[멀티라벨 전용 검증용] 각 클래스를 독립적으로 0.2에 안착시킴 (Softmax 없음)"""
    target_score = 0.2
    class_idx = [5,6,7,8,9, 15,16,17,18,19, 25,26,27,28,29]
    FIXED_TARGET_IDX = 1022 

    with torch.no_grad():
        h_center, w_center = 8 * 16, 14 * 16
        images[:, :, h_center-8:h_center+8, w_center-8:w_center+8] = 0.5 + torch.randn((1, 3, 16, 16), device=DEVICE) * 0.05
        images = torch.clamp(images, 0, 1)

    momentum = torch.zeros_like(images)
    mu = 0.9 

    for i in range(max_iters):
        images.requires_grad = True
        d32, d16 = model(images)
        
        all_cls_logits = []
        for head in [d32, d16]:
            cls = head[class_idx].view(3, 5, head.shape[1], head.shape[2])
            all_cls_logits.append(cls.permute(0, 2, 3, 1).reshape(-1, 5))
        probs_logit = torch.cat(all_cls_logits)
        
        target_P_logit = probs_logit[FIXED_TARGET_IDX]
        target_P_scores = torch.sigmoid(target_P_logit)
        
        loss = torch.mean((target_P_scores - target_score)**2)
        
        if (i+1) % 200 == 0:
            status = ", ".join([f"{p.item()*100:4.1f}%" for p in target_P_scores])
            diff = target_P_scores.max() - target_P_scores.min()
            print(f"  [ML-Exp {i+1:04d}] Max-Gap: {diff.item()*100:.1f}% | S: [{status}]")

        model.zero_grad()
        loss.backward()
        
        if images.grad is not None:
            grad = images.grad
            grad = grad / (torch.mean(torch.abs(grad)) + 1e-10)
            momentum = mu * momentum + grad
            
            with torch.no_grad():
                images -= step_size * momentum.sign() 
                images = torch.clamp(images, 0, 1)
        
        if (i+1) % 400 == 0:
            step_size *= 0.5
            
    return images.detach()

def load_and_predict_wyze(model, save_dir):
    """최종 생성된 이미지를 고정 타겟 지점(1022)에서 시그모이드 점수로 리포트"""
    transform = transforms.Compose([transforms.ToTensor()])
    image_paths = sorted([os.path.join(save_dir, img) for img in os.listdir(save_dir) if img.endswith('.png')])
    
    if not image_paths:
        return

    FIXED_TARGET_IDX = 1022
    print(f"\nEvaluating Multi-Label Challenges (Independent Sigmoids at Index {FIXED_TARGET_IDX})...")
    
    model.eval()
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
            
            # 멀티라벨 모델이므로 Sigmoid 점수를 출력
            scores = torch.sigmoid(torch.cat(p_logits)[FIXED_TARGET_IDX])
            
            prob_str = ", ".join([f"C{j}:{p*100:4.1f}%" for j, p in enumerate(scores)])
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
            # [실험] 멀티라벨 전용 최적화 함수 호출
            optimized_img = update_image_to_wyze_multi_label(model, img)
            
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
