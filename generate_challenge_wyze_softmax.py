#!/usr/bin/env python3
"""
Wyze YOLO 챌린지 생성기 — Softmax-CE 변형 (method A)
=====================================================

기존 generate_challenge_wyze.py 와의 차이:
  · Optimization loss 를 softmax cross-entropy 스타일로 교체
      - cls_logits 는 per-ROI target class 로 softmax-CE (soft target, 비포화)
      - obj_logit 은 기존 MSE 유지하되 가중치 축소 (1차 신호는 cls softmax)
  · Fingerprint 저장 포맷(.json의 rois_logits) 은 **raw logit 그대로** 유지
      → evaluate_challenges.py 그대로 재사용 가능 (BFA diff = logit 편차)

왜 target 을 one-hot 으로 쓰지 않나:
  · one-hot 을 쫓으면 target class logit 이 극단값으로 포화 → bit-flip 민감도 감소
  · soft target (예: target=0.6, 나머지=0.1) 은 비포화 영역에서 수렴 →
    logit 공간의 attestation 신호를 보존

왜 obj 는 softmax 에서 빼나:
  · obj 는 이진(objectness) 채널로 YOLO 구조상 cls softmax 와 별개 축
  · 완전 제거하면 obj 쪽 채널의 비트 플립이 fingerprint 에서 빠지므로
    낮은 가중치(OBJ_LOSS_WEIGHT)로 mild anchor 만 걸어둠

출력:
  data/challenge/wyze_ste_softmax/
    fingerprint_rois.json       — 공통 ROI 정의 (+ mode="softmax_ce")
    challenge_NNN.bin           — 640x360x3 uint8 raw frame
    challenge_NNN.json          — per-ROI raw logit + softmax_targets 메타

평가:
    python evaluate_challenges.py --save_dir data/challenge/wyze_ste_softmax
"""

import re
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WYZE_DIR   = SCRIPT_DIR / "wyze_model"
sys.path.insert(0, str(WYZE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from classify import load_model, preprocess, load_input, CAM_H, CAM_W, CLASS_NAMES

# 기존 파일에서 재사용하는 순수 helper 들 (loss 외 공통 로직).
# ROI 선택 기준은 동일 (중립 입력에서 dist_to_target 최소) → 두 방식 간
# 비교 실험 시 ROI 좌표가 같아야 의미 있는 비교가 됨.
from generate_challenge_wyze import (
    extract_roi_logits,
    max_dev_vs_stored,
    init_raw,
    save_raw_bin,
    select_golden_rois,
)


DEVICE = torch.device('cpu')

NUM_CHALLENGES      = 100
K_GOLDEN            = 16

# Softmax soft target. target_class 에 이 확률, 나머지 4개 class 는 균등 분배.
# 0.6 은 실험적으로 safe: logit gap ≈ ln(0.6/0.1) ≈ 1.79 → 포화 전 영역.
SOFTMAX_TARGET_PROB = 0.6

# Obj 는 softmax 밖 별도 채널. 기존 target 유지하되 가중치 축소.
# 완전히 빼면 obj 채널의 비트 플립이 fingerprint 민감도에서 누락됨.
OBJ_TARGET_LOGIT    = -10.0
OBJ_LOSS_WEIGHT     = 0.1

MAX_ITERS           = 2000
CONVERGENCE_DEV     = 0.05          # softmax 공간 max |achieved - target|
VERIFY_TOLERANCE    = 0.5           # logit 공간 round-trip 허용 (기존과 동일)

ADAM_LR             = 0.3
ADAM_LR_DECAY_AT    = 1000
ADAM_LR_DECAY_FACTOR = 0.3
LOG_EVERY           = 250

SAVE_DIR = SCRIPT_DIR / "data" / "challenge" / "wyze_ste_softmax"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# Per-ROI target class 할당
# ────────────────────────────────────────────────────────────────────────────

def assign_target_classes(ch_idx, num_rois, num_classes=5):
    """Challenge · ROI index 만으로 결정되는 deterministic target class.

    7, 3 은 5 와 서로 소 → ROI 간 class 분포가 골고루 퍼짐.
    Verifier 는 raw logit 만 비교하므로 target class 자체는 재현에 불필요하지만,
    실험 분석(어떤 class 를 쫓았는지) 용도로 .json 에 저장.
    """
    return [(ch_idx * 7 + i * 3) % num_classes for i in range(num_rois)]


# ────────────────────────────────────────────────────────────────────────────
# Softmax-CE Loss
# ────────────────────────────────────────────────────────────────────────────

def fingerprint_loss_softmax(d32, d16, rois, target_classes):
    """Per-ROI softmax CE (cls, soft target) + MSE (obj, low weight).

    반환:
      loss         — backward 용 스칼라 tensor
      max_prob_dev — softmax 공간에서 achieved 확률이 target 분포와 가장 먼 성분의 |차이|
                     (logging / convergence criterion 용)
    """
    heads = [d32, d16]
    loss = d32.new_zeros(())
    max_prob_dev = 0.0

    non_target_prob = (1.0 - SOFTMAX_TARGET_PROB) / 4.0

    for r, tgt in zip(rois, target_classes):
        head = heads[r['head_idx']]
        gy, gx = r['gy'], r['gx']
        obj_l = head[r['obj_ch'], gy, gx]
        cls_l = torch.stack([head[c, gy, gx] for c in r['cls_ch']])

        log_probs  = F.log_softmax(cls_l, dim=-1)
        target_vec = torch.full_like(cls_l, non_target_prob)
        target_vec[tgt] = SOFTMAX_TARGET_PROB

        # CE(soft target) = -Σ p_target * log softmax(logits). KL 과 동일 최적화.
        ce = -(target_vec * log_probs).sum()

        loss = loss + ce + OBJ_LOSS_WEIGHT * (obj_l - OBJ_TARGET_LOGIT) ** 2

        with torch.no_grad():
            probs = F.softmax(cls_l, dim=-1)
            max_prob_dev = max(max_prob_dev,
                               (probs - target_vec).abs().max().item())

    return loss, max_prob_dev


# ────────────────────────────────────────────────────────────────────────────
# Raw 이미지 최적화
# ────────────────────────────────────────────────────────────────────────────

def optimize_raw_softmax(model, raw, rois, target_classes, challenge_idx):
    """Adam + StepLR. Best-so-far 는 softmax prob_dev 기준으로 갱신.

    반환: {'raw', 'logits', 'prob_dev', 'iters', 'converged'}
    logits 는 항상 raw logit (attestation fingerprint 용).
    """
    raw = raw.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=ADAM_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=ADAM_LR_DECAY_AT, gamma=ADAM_LR_DECAY_FACTOR
    )

    best = {
        'raw': raw.detach().clone(),
        'logits': None,
        'prob_dev': float('inf'),
        'iters': 0,
    }

    for i in range(MAX_ITERS):
        optimizer.zero_grad()

        x_int8, _ = preprocess(raw)
        d32, d16 = model(x_int8.squeeze(0))
        loss, prob_dev = fingerprint_loss_softmax(d32, d16, rois, target_classes)

        if prob_dev < best['prob_dev']:
            best['prob_dev'] = prob_dev
            best['raw'] = raw.detach().clone()
            best['logits'] = extract_roi_logits(d32, d16, rois)
            best['iters'] = i + 1

        if prob_dev < CONVERGENCE_DEV:
            best['converged'] = True
            return best

        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            raw.nan_to_num_(nan=128.0, posinf=255.0, neginf=0.0)
            raw.clamp_(0.0, 255.0)

        if (i + 1) % LOG_EVERY == 0:
            print(f"      [ch {challenge_idx:03d} iter {i+1:4d}] "
                  f"loss={loss.item():8.3f}  prob_dev={prob_dev:.4f}  "
                  f"best={best['prob_dev']:.4f}")

    best['converged'] = False
    return best


# ────────────────────────────────────────────────────────────────────────────
# Save / Verify
# ────────────────────────────────────────────────────────────────────────────

def save_challenge(best, ch_idx, target_classes):
    """.bin + .json 저장. rois_logits 포맷은 multiroi 변형과 동일 → evaluator 호환."""
    bin_path  = SAVE_DIR / f"challenge_{ch_idx:03d}.bin"
    json_path = SAVE_DIR / f"challenge_{ch_idx:03d}.json"

    save_raw_bin(best['raw'], bin_path)
    with open(json_path, 'w') as f:
        json.dump({
            'challenge_idx': ch_idx,
            'mode': 'softmax_ce',
            'converged': best['converged'],
            'prob_dev': best['prob_dev'],
            'iters_to_best': best['iters'],
            'softmax_targets': {
                'target_prob': SOFTMAX_TARGET_PROB,
                'target_classes': target_classes,
            },
            'rois_logits': best['logits'],
        }, f, indent=2)
    return bin_path


def verify_all(model, rois):
    """저장된 .bin 재로딩 → forward → 저장된 raw logit 과 비교.
    evaluate_challenges.py 의 체크와 동일 로직 (self-contained 용).
    """
    print("\n=== Verification (reload .bin → forward → compare with .json) ===")
    paths = sorted(SAVE_DIR.glob("challenge_*.bin"))
    if not paths:
        print("  (no challenges to verify)")
        return

    passed, failed, skipped = 0, 0, 0
    worst = 0.0
    for p in paths:
        json_path = p.with_suffix('.json')
        if not json_path.exists():
            print(f"  [SKIP] {p.name}: per-challenge fingerprint .json 없음")
            skipped += 1
            continue

        x_int8, _ = load_input(str(p), preprocessed=False)
        with torch.no_grad():
            d32, d16 = model(x_int8.squeeze(0))
        current = extract_roi_logits(d32, d16, rois)

        with open(json_path) as f:
            stored = json.load(f)['rois_logits']

        md = max_dev_vs_stored(current, stored)
        worst = max(worst, md)
        ok = md < VERIFY_TOLERANCE
        tag = "OK  " if ok else "FAIL"
        passed += int(ok); failed += int(not ok)
        print(f"  [{tag}] {p.name:28s} round-trip max_dev={md:.6f}")

    total = passed + failed
    print(f"\n  Summary: {passed}/{total} OK  "
          f"(worst={worst:.6f}, skipped={skipped})")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print("Loading Wyze STE model via classify.load_model('ste') ...")
    model, model_type = load_model('ste')
    model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  model_type={model_type}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    print(f"\nSelecting top {K_GOLDEN} golden ROIs (neutral gray input) ...")
    rois = select_golden_rois(model, K=K_GOLDEN)
    for i, r in enumerate(rois):
        head_name = 'S32' if r['head_idx'] == 0 else 'S16'
        print(f"  [{i:02d}] {head_name} a={r['anchor']} "
              f"cell=({r['gy']:2d},{r['gx']:2d})  "
              f"dist_to_target={r['dist_to_target']:.3f}")

    fingerprint_path = SAVE_DIR / "fingerprint_rois.json"
    with open(fingerprint_path, "w") as f:
        json.dump({
            'K': K_GOLDEN,
            'mode': 'softmax_ce',
            'softmax_target_prob': SOFTMAX_TARGET_PROB,
            'obj_target_logit': OBJ_TARGET_LOGIT,
            'obj_loss_weight': OBJ_LOSS_WEIGHT,
            'convergence_dev': CONVERGENCE_DEV,
            'rois': rois,
            'class_names': CLASS_NAMES,
        }, f, indent=2)
    print(f"  fingerprint saved → {fingerprint_path.name}")

    pattern = re.compile(r"challenge_(\d+)\.bin")
    existing, max_idx = 0, 0
    for p in SAVE_DIR.iterdir():
        m = pattern.match(p.name)
        if m:
            existing += 1
            max_idx = max(max_idx, int(m.group(1)))
    remaining = max(0, NUM_CHALLENGES - existing)
    print(f"\nExisting: {existing}/{NUM_CHALLENGES}. Generating {remaining} more ...")

    if remaining == 0:
        verify_all(model, rois)
        return

    generated, converged = 0, 0
    for step in range(remaining):
        ch_idx = max_idx + step + 1
        print(f"\n  ── Challenge {ch_idx:03d} ──")
        target_classes = assign_target_classes(ch_idx, len(rois))
        raw = init_raw()
        best = optimize_raw_softmax(model, raw, rois, target_classes,
                                    challenge_idx=ch_idx)

        path = save_challenge(best, ch_idx, target_classes)
        generated += 1
        if best['converged']:
            converged += 1
            tag = "OK"
        else:
            tag = "BEST"
        print(f"  [{tag}] iters={best['iters']:4d}  "
              f"prob_dev={best['prob_dev']:.4f}  → {path.name}")

    print(f"\nGenerated {generated} challenges. Converged: {converged}/{generated}.")
    verify_all(model, rois)


if __name__ == "__main__":
    main()
