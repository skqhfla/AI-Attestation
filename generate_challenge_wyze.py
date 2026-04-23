#!/usr/bin/env python3
"""
Wyze YOLO 무결성 검증용 챌린지 생성기 (A+B)
============================================

A. 디바이스 bit-exact forward 확보
   - classify.load_model('ste') + classify.preprocess() 를 그대로 사용
   - 저장 포맷: 640x360x3 uint8 raw frame (.bin) — 디바이스/classify.py와 직접 호환

B. Fingerprint 커버리지 확대
   - 단일 ROI가 아닌 top-K "golden ROI" × (objectness + 5 class logits)
   - 기준: 중립(gray) 입력에서 logit이 포화 영역 밖(|·|<6)이면서
           목표 logit과의 거리가 가장 짧은 ROI K개 선정
           → gradient가 살아있고, 가중치의 다양한 비트를 경유함

Fingerprint target:
   obj_logit  → 0.0     (sigmoid = 0.5)
   cls_logit  → -1.3863 (sigmoid = 0.2; 5-class multi-label 균형점)
"""

import re
import sys
import json
import numpy as np
import torch
from pathlib import Path

# Wyze 모델 의존성은 AI-Attestation/wyze_model/ 에 vendor 되어 있음
# (classify.py, model_torch_ste.py, model_blob, weight_blob, libstb_resize.so)
# libstb_resize.so 는 Linux x86-64 ELF. 다른 OS 에서는 stb_image_resize.h 로 재빌드 필요.
SCRIPT_DIR = Path(__file__).resolve().parent
WYZE_DIR   = SCRIPT_DIR / "wyze_model"
sys.path.insert(0, str(WYZE_DIR))

from classify import load_model, preprocess, load_input, CAM_H, CAM_W, CLASS_NAMES


# preprocess() 의 bit-exact 경로가 numpy/ctypes 기반이라 CPU 에서 구동.
# (CUDA 텐서를 섞으면 device mismatch 발생)
DEVICE = torch.device('cpu')

NUM_CHALLENGES      = 100
K_GOLDEN            = 16

# 관찰: neutral gray 입력에서 INT8 YOLO 의 logit 분포는
#   obj_logit  ~ [-14.8, -8.9]  (mean -11.9)
#   cls_logit  ~ [-6.1,   1.0]  (mean -3.0)
# Attestation 관점에서 target 값 자체는 의미 없고 (고정·재현만 되면 됨),
# reachable 영역 내로 잡아야 수렴이 빠름. 실제 logit 분포 중앙 근처로 설정.
OBJ_TARGET_LOGIT    = -10.0
CLS_TARGET_LOGIT    = -3.0

MAX_ITERS           = 2000
CONVERGENCE_DEV     = 0.5           # max |logit - target| 기준
VERIFY_TOLERANCE    = 1.0           # 재로딩 후 허용 편차

# Adam on raw pixels(0~255).
# sign-PGD는 L∞ bound 용이라 MSE loss 의 미세 조정에서 진동. Adam 의 2차 모멘트
# 스케일링으로 수렴 근처 자동 감속 → 안정적으로 local min 도달.
ADAM_LR             = 1.0
LOG_EVERY           = 250

SAVE_DIR = SCRIPT_DIR / "data" / "challenge" / "wyze_ste_multiroi"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# Golden ROI 선정 (중립 입력 한 번만 forward)
# ────────────────────────────────────────────────────────────────────────────

def select_golden_rois(model, K):
    """중립 gray(128) 입력에서 목표 logit과 거리(dist_to_target)가 가장 짧은 ROI K개.

    INT8 양자화 YOLO 는 중립 입력에서 obj logit이 큰 음수 영역(-20 ~ -30)에 몰려
    있는 경우가 많아 saturation 필터는 적용하지 않음. STE backward 가 trunc/round
    를 우회해 identity로 전파되므로 포화 영역이어도 gradient 가 살아있음.

    score = (obj_logit - OBJ_TARGET)^2 + sum((cls_logit - CLS_TARGET)^2)
    """
    neutral = torch.full((1, 3, CAM_H, CAM_W), 128.0)
    with torch.no_grad():
        x, _ = preprocess(neutral)
        d32, d16 = model(x.squeeze(0))

    heads = [(d32, 0, 8, 14), (d16, 1, 16, 28)]
    candidates = []
    obj_stats, cls_stats = [], []
    for head, head_idx, gh, gw in heads:
        for a in range(3):
            obj_ch = a * 10 + 4
            cls_ch = [a * 10 + 5 + c for c in range(5)]
            for gy in range(gh):
                for gx in range(gw):
                    obj_l = float(head[obj_ch, gy, gx])
                    cls_l = [float(head[c, gy, gx]) for c in cls_ch]
                    obj_stats.append(obj_l)
                    cls_stats.extend(cls_l)
                    dist = (obj_l - OBJ_TARGET_LOGIT) ** 2 + sum(
                        (l - CLS_TARGET_LOGIT) ** 2 for l in cls_l
                    )
                    candidates.append({
                        'head_idx': head_idx,
                        'anchor': a,
                        'gy': gy,
                        'gx': gx,
                        'obj_ch': obj_ch,
                        'cls_ch': cls_ch,
                        'dist_to_target': dist,
                        'obj_logit_init': obj_l,
                        'cls_logits_init': cls_l,
                    })

    # 진단 출력: 중립 입력에서의 logit 분포
    obj_arr = np.array(obj_stats)
    cls_arr = np.array(cls_stats)
    print(f"  [diag] obj_logit  min={obj_arr.min():7.2f} max={obj_arr.max():7.2f} "
          f"mean={obj_arr.mean():7.2f}")
    print(f"  [diag] cls_logit  min={cls_arr.min():7.2f} max={cls_arr.max():7.2f} "
          f"mean={cls_arr.mean():7.2f}")

    candidates.sort(key=lambda r: r['dist_to_target'])
    return candidates[:K]


# ────────────────────────────────────────────────────────────────────────────
# Fingerprint loss
# ────────────────────────────────────────────────────────────────────────────

def fingerprint_loss(d32, d16, rois):
    """K ROI × 6 logit (obj + 5 cls) 에 대한 MSE + max deviation 반환."""
    heads = [d32, d16]
    loss = d32.new_zeros(())
    obj_dev_max = 0.0
    cls_dev_max = 0.0

    for r in rois:
        head = heads[r['head_idx']]
        gy, gx = r['gy'], r['gx']
        obj_l = head[r['obj_ch'], gy, gx]
        cls_l = torch.stack([head[c, gy, gx] for c in r['cls_ch']])

        loss = loss + (obj_l - OBJ_TARGET_LOGIT) ** 2
        loss = loss + ((cls_l - CLS_TARGET_LOGIT) ** 2).sum()

        obj_dev_max = max(obj_dev_max, abs(obj_l.item() - OBJ_TARGET_LOGIT))
        cls_dev_max = max(cls_dev_max,
                          (cls_l.detach() - CLS_TARGET_LOGIT).abs().max().item())

    max_dev = max(obj_dev_max, cls_dev_max)
    return loss, max_dev


# ────────────────────────────────────────────────────────────────────────────
# Raw 이미지 생성 / 최적화
# ────────────────────────────────────────────────────────────────────────────

def init_raw():
    """640x360x3, [0,255] 범위. uniform / clipped gaussian 섞어서 다양성 확보."""
    if np.random.rand() > 0.5:
        return torch.rand((1, 3, CAM_H, CAM_W), device=DEVICE) * 255.0
    g = torch.randn((1, 3, CAM_H, CAM_W), device=DEVICE) * 30.0 + 128.0
    return torch.clamp(g, 0.0, 255.0)


def optimize_raw(model, raw, rois, challenge_idx):
    """Adam + gradient descent on raw pixels (0~255). Clamp to [0,255] per step.

    gradient 는 preprocess 의 미분가능 근사를 통해 raw(640x360 uint8) 까지 전파.
    forward 는 stb_image_resize 로 디바이스와 bit-exact.
    Adam 의 2차 모멘트 스케일링으로 수렴 근처에서 자동 감속 (sign-PGD 진동 회피).
    """
    raw = raw.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=ADAM_LR)

    last_max_dev = float('inf')
    for i in range(MAX_ITERS):
        optimizer.zero_grad()

        x_int8, _ = preprocess(raw)
        d32, d16 = model(x_int8.squeeze(0))
        loss, max_dev = fingerprint_loss(d32, d16, rois)
        last_max_dev = max_dev

        if max_dev < CONVERGENCE_DEV:
            return raw.detach(), True, i + 1, max_dev

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            raw.clamp_(0.0, 255.0)

        if (i + 1) % LOG_EVERY == 0:
            print(f"      [ch {challenge_idx:03d} iter {i+1:4d}] "
                  f"loss={loss.item():8.2f}  max_dev={max_dev:5.3f}")

    return raw.detach(), False, MAX_ITERS, last_max_dev


def save_raw_bin(raw, path):
    """640x360x3 uint8 raw frame 으로 저장. classify.load_input() 으로 재현 가능."""
    arr = raw.squeeze(0).detach().cpu().numpy()          # (3, 360, 640)
    arr = arr.transpose(1, 2, 0)                         # (360, 640, 3)
    arr_u8 = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    arr_u8.tofile(str(path))


# ────────────────────────────────────────────────────────────────────────────
# Verification: 저장된 .bin → load_input → preprocess → model → fingerprint 확인
# ────────────────────────────────────────────────────────────────────────────

def verify_all(model, rois):
    print("\n=== Verification (classify.load_input → preprocess → model) ===")
    paths = sorted(SAVE_DIR.glob("challenge_*.bin"))
    if not paths:
        print("  (no challenges to verify)")
        return

    passed, failed = 0, 0
    worst = 0.0
    for p in paths:
        x_int8, _ = load_input(str(p), preprocessed=False)
        with torch.no_grad():
            d32, d16 = model(x_int8.squeeze(0))
        _, max_dev = fingerprint_loss(d32, d16, rois)
        worst = max(worst, max_dev)
        ok = max_dev < VERIFY_TOLERANCE
        tag = "OK  " if ok else "FAIL"
        passed += int(ok); failed += int(not ok)
        print(f"  [{tag}] {p.name:28s} max_dev={max_dev:.4f}")

    print(f"\n  Summary: {passed}/{passed+failed} OK  (worst max_dev={worst:.4f})")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print("Loading Wyze STE model via classify.load_model('ste') ...")
    model, model_type = load_model('ste')
    model.to(DEVICE)
    model.eval()
    # 모델 파라미터는 gradient 불필요 (raw 만 최적화 대상)
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
            'obj_target_logit': OBJ_TARGET_LOGIT,
            'cls_target_logit': CLS_TARGET_LOGIT,
            'convergence_dev': CONVERGENCE_DEV,
            'rois': rois,
            'class_names': CLASS_NAMES,
        }, f, indent=2)
    print(f"  fingerprint saved → {fingerprint_path.name}")

    # 기존 챌린지 인덱싱
    pattern = re.compile(r"challenge_(\d+)\.bin")
    existing = 0
    max_idx = 0
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

    generated = 0
    attempts = 0
    while generated < remaining:
        attempts += 1
        ch_idx = max_idx + generated + 1
        print(f"\n  ── Challenge {ch_idx:03d}  (attempt {attempts}) ──")
        raw = init_raw()
        optimized, success, iters, max_dev = optimize_raw(
            model, raw, rois, challenge_idx=ch_idx
        )

        if success:
            path = SAVE_DIR / f"challenge_{ch_idx:03d}.bin"
            save_raw_bin(optimized, path)
            generated += 1
            print(f"  [OK] converged in {iters} iters, max_dev={max_dev:.4f}  → {path.name}")
        else:
            print(f"  [retry] did NOT converge (max_dev={max_dev:.4f}, iters={iters})")

    print(f"\nGenerated {generated} challenges in {attempts} attempt(s).")
    verify_all(model, rois)


if __name__ == "__main__":
    main()
