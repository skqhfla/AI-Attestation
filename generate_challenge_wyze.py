#!/usr/bin/env python3
"""
Wyze YOLO 무결성 검증용 챌린지 생성기 (A+B)
============================================

A. 디바이스 bit-exact forward 확보
   - classify.load_model('ste') + classify.preprocess() 를 그대로 사용
   - 저장 포맷: 640x360x3 uint8 raw frame (.bin) — 디바이스/classify.py와 직접 호환

B. Fingerprint 커버리지 확대
   - 단일 ROI가 아닌 top-K "golden ROI" × (objectness + 5 class logits)
   - dist_to_target 최소인 ROI K개 선정 (saturation 필터 없음)

Per-challenge fingerprint (attestation 표준 관행):
   - Optimization 은 global target (obj=-10, cls=-3) 을 쫓지만 엄격히 도달 요구 X
   - 각 challenge 의 best-so-far state 저장 + 그 때의 실제 logit 값을
     challenge_NNN.json 에 기록
   - Verifier 는 .bin 재로딩 → forward → .json 의 logit 과 비교
     (round-trip bit-exact 기대, max_dev ≈ 0)

출력 파일:
   data/challenge/wyze_ste_multiroi/
     fingerprint_rois.json       — 전체 공통: 선정된 ROI 위치/채널
     challenge_NNN.bin           — 640x360x3 uint8 raw frame
     challenge_NNN.json          — 해당 challenge 의 observed logit 값
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
CONVERGENCE_DEV     = 0.5           # quality indicator. 못 넘어도 best-so-far 저장.
VERIFY_TOLERANCE    = 0.5           # .bin round-trip 재현성 (bit-exact 기대)

# Adam on raw pixels(0~255). lr=1.0 은 raw pixel 스케일에 과도해 수렴 근처 진동.
# lr=0.3 + StepLR 로 1000 iter 후 감속 → 안정적 수렴.
ADAM_LR             = 0.3
ADAM_LR_DECAY_AT    = 1000
ADAM_LR_DECAY_FACTOR = 0.3
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


def extract_roi_logits(d32, d16, rois):
    """각 ROI 에서 obj(1) + cls(5) logit 추출. Python 리스트로 반환 (JSON 직렬화용)."""
    heads = [d32, d16]
    out = []
    for r in rois:
        head = heads[r['head_idx']]
        gy, gx = r['gy'], r['gx']
        out.append({
            'obj': float(head[r['obj_ch'], gy, gx]),
            'cls': [float(head[c, gy, gx]) for c in r['cls_ch']],
        })
    return out


def max_dev_vs_stored(current, stored):
    """current(raw 재로딩 후 forward) 와 저장된 per-challenge fingerprint 의 최대 편차."""
    md = 0.0
    for c, s in zip(current, stored):
        md = max(md, abs(c['obj'] - s['obj']))
        for cc, ss in zip(c['cls'], s['cls']):
            md = max(md, abs(cc - ss))
    return md


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
    """Adam + StepLR scheduler. Best-so-far state 추적 → 진동해도 유실 없음.

    반환: {'raw': best_raw, 'logits': best_logits, 'max_dev': best_max_dev,
           'converged': bool, 'iters': int}
    """
    raw = raw.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=ADAM_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=ADAM_LR_DECAY_AT, gamma=ADAM_LR_DECAY_FACTOR
    )

    best = {
        'raw': raw.detach().clone(),
        'logits': None,
        'max_dev': float('inf'),
        'iters': 0,
    }

    for i in range(MAX_ITERS):
        optimizer.zero_grad()

        x_int8, _ = preprocess(raw)
        d32, d16 = model(x_int8.squeeze(0))
        loss, max_dev = fingerprint_loss(d32, d16, rois)

        # best-so-far 갱신
        if max_dev < best['max_dev']:
            best['max_dev'] = max_dev
            best['raw'] = raw.detach().clone()
            best['logits'] = extract_roi_logits(d32, d16, rois)
            best['iters'] = i + 1

        if max_dev < CONVERGENCE_DEV:
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
                  f"loss={loss.item():8.2f}  max_dev={max_dev:5.3f}  "
                  f"best={best['max_dev']:5.3f}")

    best['converged'] = False
    return best


def save_raw_bin(raw, path):
    """640x360x3 uint8 raw frame 으로 저장. classify.load_input() 으로 재현 가능."""
    arr = raw.squeeze(0).detach().cpu().numpy()          # (3, 360, 640)
    arr = arr.transpose(1, 2, 0)                         # (360, 640, 3)
    arr = np.nan_to_num(arr, nan=128.0, posinf=255.0, neginf=0.0)
    arr_u8 = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    arr_u8.tofile(str(path))


def save_challenge(best, ch_idx):
    """.bin (raw image) + .json (per-challenge fingerprint) 저장.

    .json 은 해당 challenge 가 실제로 만들어낸 logit 값을 기록한다. Verifier 는
    이 값과 재로딩 후 forward 결과를 비교해 bit-flip 감지.
    """
    bin_path  = SAVE_DIR / f"challenge_{ch_idx:03d}.bin"
    json_path = SAVE_DIR / f"challenge_{ch_idx:03d}.json"

    save_raw_bin(best['raw'], bin_path)
    with open(json_path, 'w') as f:
        json.dump({
            'challenge_idx': ch_idx,
            'converged': best['converged'],
            'max_dev_vs_global_target': best['max_dev'],
            'iters_to_best': best['iters'],
            'rois_logits': best['logits'],  # list of {obj, cls[5]} per golden ROI
        }, f, indent=2)
    return bin_path


# ────────────────────────────────────────────────────────────────────────────
# Verification: 저장된 .bin → load_input → preprocess → model → fingerprint 확인
# ────────────────────────────────────────────────────────────────────────────

def verify_all(model, rois):
    """Per-challenge .json 과 재로딩 forward 결과를 비교.

    round-trip 이 bit-exact 라면 max_dev ≈ 0. VERIFY_TOLERANCE 를 넘으면
    저장/로딩 과정에서 손실 발생 or 모델 가중치 변경 의심.
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

    # 1 attempt per challenge. best-so-far 를 항상 저장 (retry 루프 제거).
    # per-challenge fingerprint 가 실제 observed logit 을 기록하므로, target 에
    # 정확히 도달 못 해도 attestation 목적상 문제 없음.
    generated = 0
    below_threshold = 0
    for step in range(remaining):
        ch_idx = max_idx + step + 1
        print(f"\n  ── Challenge {ch_idx:03d} ──")
        raw = init_raw()
        best = optimize_raw(model, raw, rois, challenge_idx=ch_idx)

        path = save_challenge(best, ch_idx)
        generated += 1
        if best['converged']:
            below_threshold += 1
            tag = "OK"
        else:
            tag = "BEST"
        print(f"  [{tag}] iters={best['iters']:4d}  max_dev={best['max_dev']:.4f}  → {path.name}")

    print(f"\nGenerated {generated} challenges. "
          f"Target-converged: {below_threshold}/{generated}.")
    verify_all(model, rois)


if __name__ == "__main__":
    main()
