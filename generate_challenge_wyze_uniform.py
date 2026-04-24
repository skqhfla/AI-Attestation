#!/usr/bin/env python3
"""
Wyze YOLO Challenge 생성기 — 단일 ROI · 균등 softmax (method U)
=================================================================

목표:
  "감지된 단 하나의 object" 에 한정해서
    · objectness 는 켜짐 상태 (sigmoid ≈ 0.95, 포화 전 영역)
    · 해당 ROI 의 5-class softmax 가 가능한 한 균등(0.2 each) 에 근접
  하도록 입력 raw frame 을 최적화.

§3 의 6 가지 장애물 중:
  (i)  softmax 축의 다중성           → ROI 1 개만 보므로 사라짐
  (ii) obj 가 softmax 밖에 있음       → obj 를 명시적 양수로 밀어 "항상 켜짐" 가정 충족
  (iv) 입력 자유도 vs ROI 간 간섭     → ROI 가 하나라 상쇄 문제 없음
 남는 건 (iii) INT8 격자, (v) dual-path, (vi) class bias 뿐 →
 "완벽한 20% 씩" 대신 "근사 균등 (19~21% 수준)" 을 현실적 목표로 함.

모델 분리 (head 주입) 은 수행하지 않음. 전체 pipeline 을 그대로 통과시키고
단지 **loss 가 ROI 1 곳만 보도록** 바꾸는 것이 이 파일의 핵심.

Loss:
  L = Var(cls_logits_at_target_ROI)                             # 5 class 서로 같게
    + OBJ_WEIGHT * (obj_logit_at_target_ROI - OBJ_ON_TARGET)^2  # obj 켜짐 (비포화)

Fingerprint 저장 전략:
  · fingerprint_rois.json: 시도 B 와 동일 기준으로 top-K=16 ROI 저장.
  · 각 challenge 는 그 중 **하나**(ch_idx mod K) 를 타깃 ROI 로 지정해 최적화.
  · 나머지 15 ROI 는 최적화에 참여하지 않지만, 관측 logit 을 challenge_NNN.json 에
    함께 기록 → 지문 다양성/커버리지 그대로 유지, evaluate_challenges.py 호환.
  · 추가 기록: target_roi_idx, 관측 softmax, uniform_gap, obj_sigmoid.

평가:
    python evaluate_challenges.py --save_dir data/challenge/wyze_ste_uniform
"""

import re
import sys
import json
import numpy as np
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WYZE_DIR   = SCRIPT_DIR / "wyze_model"
sys.path.insert(0, str(WYZE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from classify import load_model, preprocess, load_input, CAM_H, CAM_W, CLASS_NAMES

# ROI 선정/지문 추출/저장 유틸은 multiroi 구현과 공용.
# 같은 ROI 집합을 써야 시도 B/C/U 간 비교 실험이 의미가 있다.
from generate_challenge_wyze import (
    extract_roi_logits,
    max_dev_vs_stored,
    init_raw,
    save_raw_bin,
    select_golden_rois,
)


DEVICE = torch.device('cpu')

NUM_CHALLENGES = 100
K_GOLDEN       = 16            # fingerprint ROI pool (passive 관측용 포함).

# obj_logit 양수 target. sigmoid(+3.0)=0.953 → "켜짐" 은 명확하지만 포화 직전은 아님.
# +5 이상이면 sigmoid 가 0.99+ 로 포화해 gradient 가 소실됨 → +3 을 기본값으로 채택.
OBJ_ON_TARGET = 3.0

# variance loss 가 1차 목표. obj 는 상수 기준 anchor 역할이라 가중치 낮게.
OBJ_WEIGHT    = 0.1

MAX_ITERS        = 2000
# 수렴 판정: softmax 5 확률 중 (max − min). 완벽 균등=0.
# INT8 격자 한계로 현실적 하한이 0.02~0.03 부근.
CONVERGENCE_GAP        = 0.03
CONVERGENCE_OBJ_SIGMA  = 0.9   # obj 도 충분히 켜진 상태(>=0.9) 여야 수렴 인정.

VERIFY_TOLERANCE = 0.5

ADAM_LR              = 0.3
ADAM_LR_DECAY_AT     = 1000
ADAM_LR_DECAY_FACTOR = 0.3
LOG_EVERY            = 250

SAVE_DIR = SCRIPT_DIR / "data" / "challenge" / "wyze_ste_uniform"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# Loss
# ────────────────────────────────────────────────────────────────────────────

def uniform_single_roi_loss(d32, d16, rois, target_idx):
    """Target ROI 한 곳의 (Var(cls) + w·(obj−OBJ_ON)^2).

    반환:
      loss         — backward 대상 스칼라 tensor
      uniform_gap  — softmax 공간의 (max_prob − min_prob), 0 에 가까울수록 균등
      obj_sigma    — sigmoid(obj_logit) 현재값 (obj 가 실제로 켜졌는지 확인용)
    """
    heads = [d32, d16]
    r = rois[target_idx]
    head = heads[r['head_idx']]
    gy, gx = r['gy'], r['gx']

    obj_l = head[r['obj_ch'], gy, gx]
    cls_l = torch.stack([head[c, gy, gx] for c in r['cls_ch']])

    # 5 class logit 이 서로 같아지면 variance → 0 → softmax 균등.
    # unbiased=False: 표본 5개로 N 나누는 분산 (편향 수정 불필요, 최적화만 하면 됨).
    var_loss = cls_l.var(unbiased=False)
    obj_loss = (obj_l - OBJ_ON_TARGET) ** 2
    loss = var_loss + OBJ_WEIGHT * obj_loss

    with torch.no_grad():
        probs = torch.softmax(cls_l, dim=-1)
        uniform_gap = (probs.max() - probs.min()).item()
        obj_sigma = torch.sigmoid(obj_l).item()

    return loss, uniform_gap, obj_sigma


# ────────────────────────────────────────────────────────────────────────────
# Optimization
# ────────────────────────────────────────────────────────────────────────────

def optimize_raw_uniform(model, raw, rois, target_idx, challenge_idx):
    """Adam + StepLR. Best-so-far 는 (obj 가 켜진 상태에서의) uniform_gap 기준.

    obj 가 꺼진 채(sigmoid<0.5) 우연히 cls 가 균등해 보이는 "가짜 수렴" 을 막기 위해
    best 업데이트 조건에 obj_sigma > 0.5 을 요구.

    반환: {'raw', 'logits'(K 개 전체), 'uniform_gap', 'obj_sigma',
           'target_softmax', 'iters', 'converged'}
    """
    raw = raw.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=ADAM_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=ADAM_LR_DECAY_AT, gamma=ADAM_LR_DECAY_FACTOR
    )

    best = {
        'raw': raw.detach().clone(),
        'logits': None,
        'uniform_gap': float('inf'),
        'obj_sigma': 0.0,
        'target_softmax': None,
        'iters': 0,
    }

    for i in range(MAX_ITERS):
        optimizer.zero_grad()

        x_int8, _ = preprocess(raw)
        d32, d16 = model(x_int8.squeeze(0))
        loss, uniform_gap, obj_sigma = uniform_single_roi_loss(
            d32, d16, rois, target_idx
        )

        # best-so-far 갱신: obj 가 켜져 있고(>0.5), uniform_gap 이 더 작을 때.
        if obj_sigma > 0.5 and uniform_gap < best['uniform_gap']:
            best['uniform_gap'] = uniform_gap
            best['obj_sigma']   = obj_sigma
            best['raw']         = raw.detach().clone()
            best['logits']      = extract_roi_logits(d32, d16, rois)
            best['iters']       = i + 1

            # target ROI 의 observed softmax 기록 (attestation 참고치, 비교용)
            r = rois[target_idx]
            head = [d32, d16][r['head_idx']]
            with torch.no_grad():
                cls_l = torch.stack([head[c, r['gy'], r['gx']] for c in r['cls_ch']])
                best['target_softmax'] = torch.softmax(cls_l, dim=-1).tolist()

        if (uniform_gap < CONVERGENCE_GAP and
                obj_sigma > CONVERGENCE_OBJ_SIGMA):
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
                  f"loss={loss.item():7.3f}  gap={uniform_gap:.4f}  "
                  f"obj_σ={obj_sigma:.3f}  "
                  f"best_gap={best['uniform_gap']:.4f}")

    best['converged'] = False
    return best


# ────────────────────────────────────────────────────────────────────────────
# Save / Verify
# ────────────────────────────────────────────────────────────────────────────

def save_challenge(best, ch_idx, target_idx, rois):
    """.bin + .json 저장. rois_logits 포맷은 multiroi 변형과 동일 → evaluator 호환."""
    bin_path  = SAVE_DIR / f"challenge_{ch_idx:03d}.bin"
    json_path = SAVE_DIR / f"challenge_{ch_idx:03d}.json"

    save_raw_bin(best['raw'], bin_path)

    r = rois[target_idx]
    with open(json_path, 'w') as f:
        json.dump({
            'challenge_idx': ch_idx,
            'mode': 'uniform_single_roi',
            'converged': best['converged'],
            'target_roi_idx': target_idx,
            'target_roi_cell': {
                'head': 'S32' if r['head_idx'] == 0 else 'S16',
                'anchor': r['anchor'],
                'gy': r['gy'], 'gx': r['gx'],
            },
            'uniform_gap': best['uniform_gap'],
            'obj_sigmoid_at_target': best['obj_sigma'],
            'observed_softmax_at_target': best['target_softmax'],
            'iters_to_best': best['iters'],
            'rois_logits': best['logits'],   # K 개 전체 관측값. evaluator 는 그대로 사용.
        }, f, indent=2)
    return bin_path


def verify_all(model, rois):
    """저장된 .bin 재로딩 → forward → 저장 logit 과 비교.
    evaluate_challenges.py 의 로직과 동일(자체 검증용)."""
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
            'mode': 'uniform_single_roi',
            'obj_on_target': OBJ_ON_TARGET,
            'obj_weight': OBJ_WEIGHT,
            'convergence_gap': CONVERGENCE_GAP,
            'convergence_obj_sigma': CONVERGENCE_OBJ_SIGMA,
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
    gaps = []
    for step in range(remaining):
        ch_idx = max_idx + step + 1
        # challenge 마다 타깃 ROI 를 cyclic 으로 바꿔 지문 다양성 확보.
        target_idx = (ch_idx - 1) % K_GOLDEN

        r = rois[target_idx]
        head_name = 'S32' if r['head_idx'] == 0 else 'S16'
        print(f"\n  ── Challenge {ch_idx:03d} "
              f"(target ROI [{target_idx:02d}] {head_name} "
              f"a={r['anchor']} cell=({r['gy']},{r['gx']})) ──")

        raw = init_raw()
        best = optimize_raw_uniform(model, raw, rois, target_idx,
                                    challenge_idx=ch_idx)

        path = save_challenge(best, ch_idx, target_idx, rois)
        generated += 1
        gaps.append(best['uniform_gap'])
        tag = "OK" if best['converged'] else "BEST"
        sm = best['target_softmax']
        sm_str = "[" + ", ".join(f"{p:.3f}" for p in sm) + "]" if sm else "N/A"
        print(f"  [{tag}] iters={best['iters']:4d}  "
              f"gap={best['uniform_gap']:.4f}  obj_σ={best['obj_sigma']:.3f}  "
              f"softmax={sm_str}  → {path.name}")

    if gaps:
        gaps_arr = np.array(gaps)
        print(f"\nGenerated {generated} challenges. "
              f"Converged (gap<{CONVERGENCE_GAP}, obj>{CONVERGENCE_OBJ_SIGMA}): "
              f"{sum(1 for g in gaps if g < CONVERGENCE_GAP)}/{generated}.")
        print(f"  uniform_gap   mean={gaps_arr.mean():.4f}  "
              f"median={np.median(gaps_arr):.4f}  "
              f"min={gaps_arr.min():.4f}  max={gaps_arr.max():.4f}")
    verify_all(model, rois)


if __name__ == "__main__":
    main()
