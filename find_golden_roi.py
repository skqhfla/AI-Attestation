import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# 경로 설정
CURRENT_DIR = Path("C:/Users/skqhf/Documents/카카오톡 받은 파일/extracted_model/AI-Attestation")
sys.path.append(str(CURRENT_DIR))

from models.quan_wyze import wyze_resnet20_quan

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def scan_model():
    print(f"Loading Wyze model on {DEVICE}...")
    model = wyze_resnet20_quan().to(DEVICE)
    model.eval()

    class_idx = [5,6,7,8,9, 15,16,17,18,19, 25,26,27,28,29]
    CLASS_NAMES = ["person", "vehicle", "pet", "package", "face"]

    # 1. 0.5 중간값 이미지로 대포주사 수행
    input_img = torch.full((1, 3, 256, 448), 0.5, device=DEVICE)
    
    with torch.no_grad():
        d32, d16 = model(input_img)
        
        all_logits = []
        for head in [d32, d16]:
            cls = head[class_idx].view(3, 5, head.shape[1], head.shape[2])
            all_logits.append(cls.permute(0, 2, 3, 1).reshape(-1, 5))
        
        full_logits = torch.cat(all_logits) # (1680, 5)
        
    print(f"Scanning {len(full_logits)} ROIs for Golden conditions...")
    
    results = []
    for idx in range(len(full_logits)):
        logits = full_logits[idx]
        scores = torch.sigmoid(logits)
        
        face_logit = logits[4].item() # Face 감도
        person_logit = logits[0].item()
        
        # 필터링 조건 1: 얼굴 로짓이 너무 낮지 않아야 함 (Dead zone 제외)
        if face_logit < -6.5:
            continue
            
        # 평가 지표: 로짓의 평균과 분산
        mean_logit = torch.mean(logits).item()
        var_logit = torch.var(logits).item()
        
        # 황금 점수: (얼굴 감도 높을수록) + (분산 낮을수록)
        # 분산에 5배 가중치를 두어 클래스 균형을 강조
        golden_score = face_logit - (5.0 * var_logit)
        
        results.append({
            'idx': idx,
            'logits': logits.cpu().numpy(),
            'scores': scores.cpu().numpy() * 100,
            'golden_score': golden_score,
            'face_logit': face_logit
        })

    # Golden Score 기준 정렬 (높은 순)
    results.sort(key=lambda x: x['golden_score'], reverse=True)

    print("\n--- TOP 10 GOLDEN ROI CANDIDATES ---")
    print(f"{'Rank':<5} {'IDX':<6} {'Score':<8} {'FaceLog':<8} {'Scores (%)':<40}")
    for i, r in enumerate(results[:10]):
        s_str = ", ".join([f"{s:4.1f}" for s in r['scores']])
        print(f"{i+1:<5} {r['idx']:<6} {r['golden_score']:<8.2f} {r['face_logit']:<8.2f} [{s_str}]")

    return results[0]['idx'] if results else None

if __name__ == "__main__":
    scan_model()

