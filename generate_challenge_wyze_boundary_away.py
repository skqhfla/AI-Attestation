#!/usr/bin/env python3
"""
Wyze YOLO 챌린지 생성기 — Decision-boundary-AWAY (편중 랜덤, method R+)
=====================================================================

동기
----
generate_challenge_wyze_uniform.py 가 만드는 decision-boundary-AWARE 챌린지는
target ROI 의 5-class softmax 를 **균등**(top1≈top2, margin≈0)하게 만들어
경계 위에 세운다 → BFA(비트 플립)로 조금만 흔들려도 Top-k 순위가 바뀌어
탐지율이 높다.

이 파일은 그 **반대**를 만든다: class 벡터가 **한 클래스로 강하게 쏠린**
(top1 − top2 margin 이 큰 = 경계에서 멀리 떨어진) 입력을 고른다. 이런 입력은
비트 플립에도 Top-k 순위가 잘 안 바뀌므로 탐지율이 낮아야 하며, 이는
"경계에 세우는 것이 왜 중요한가" 를 보여주는 baseline 이 된다.
(§5.3 Comparison with Random Challenges 확장: 순수 random 보다 더 확실히
 boundary 에서 먼 baseline.)

핵심: **최적화하지 않는다.** generate_random_inputs.py 처럼 순수 랜덤 입력을
대량 생성하되, target ROI 에서 class 벡터가 가장 쏠린 후보만 **선별(rejection
sampling)** 한다. 따라서 여전히 "랜덤 챌린지" 성격을 유지한다.

방식 (사용자 확정 사항)
----------------------
  · 편중 방식 : rejection sampling (선별). 입력을 최적화하지 않음.
  · 타깃 클래스: 입력별 argmax (자연 최댓값). 고정 클래스 아님.
  · 점수      : score='margin' → top1 − top2 (경계로부터의 거리, 기본값)
                score='top1'   → top1 logit 값 자체
  · target ROI: challenge 마다 cyclic ((ch_idx-1) % K) — uniform 생성기 및
                compute_detection_rate.py 의 target_roi_only 단위와 정합.

출력 (generate_random_inputs.py / evaluate_challenges.py 와 100% 호환)
--------------------------------------------------------------------
  data/challenge/wyze_boundary_away/
    fingerprint_rois.json   — 공통 ROI 정의 (boundary-aware 세트와 동일 ROI 재사용)
    challenge_NNN.bin       — 640x360x3 uint8 raw frame
    challenge_NNN.json      — rois_logits / rois_topk (+ boundary-away 메타)

사용
----
  python generate_challenge_wyze_boundary_away.py
  python generate_challenge_wyze_boundary_away.py --num 100 --pool 300
  python generate_challenge_wyze_boundary_away.py --score top1 --seed 777

평가
----
  python evaluate_challenges.py --save_dir data/challenge/wyze_boundary_away
  python evaluate_challenges.py --save_dir data/challenge/wyze_boundary_away \\
      --weights <bfa.pth> --output post.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
WYZE_DIR   = SCRIPT_DIR / "wyze_model"
sys.path.insert(0, str(WYZE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from classify import load_model, preprocess, load_input, CLASS_NAMES  # noqa: E402

# ROI 선정 / 입력 초기화 / bin 저장 / 지문 추출 유틸은 challenge 생성기와 공용.
# 같은 ROI 집합을 써야 boundary-aware vs boundary-away 비교가 의미 있다.
from generate_challenge_wyze import (                                  # noqa: E402
    extract_roi_logits,
    extract_roi_topk,
    init_raw,
    save_raw_bin,
    select_golden_rois,
)

DEVICE = torch.device("cpu")

# boundary-aware(uniform) 세트와 동일하게 K=16 ROI 사용.
K_GOLDEN = 16

DEFAULT_SAVE_DIR = SCRIPT_DIR / "data" / "challenge" / "wyze_boundary_away"

# challenge 생성 때 만들어둔 fingerprint 가 있으면 그대로 재사용해
# 두 방식이 완전히 동일한 ROI 좌표를 쓰게 한다 (공정 비교).
DEFAULT_FINGERPRINT = (
    SCRIPT_DIR / "data" / "challenge" / "wyze_ste_uniform" / "fingerprint_rois.json"
)


# ────────────────────────────────────────────────────────────────────────────
# 후보 점수: target ROI 의 class 벡터가 얼마나 한쪽으로 쏠렸는가
# ────────────────────────────────────────────────────────────────────────────

def roi_cls_logits(d32, d16, roi):
    """주어진 ROI 의 5-class logit 벡터(list[float])."""
    head = [d32, d16][roi["head_idx"]]
    gy, gx = roi["gy"], roi["gx"]
    return [float(head[c, gy, gx]) for c in roi["cls_ch"]]


def skew_score(cls_logits, mode):
    """class 벡터의 '쏠림' 점수. 클수록 경계에서 멀다(=boundary-away).

    mode='margin' : top1 − top2  (decision boundary 로부터의 거리)
    mode='top1'   : top1 logit 값 자체 (가장 큰 클래스 벡터 성분)

    반환: (score, argmax_class_idx, sorted_desc_logits)
    """
    order = sorted(range(len(cls_logits)), key=lambda i: (-cls_logits[i], i))
    top1_idx = order[0]
    top1 = cls_logits[top1_idx]
    top2 = cls_logits[order[1]] if len(order) > 1 else top1
    score = (top1 - top2) if mode == "margin" else top1
    return score, top1_idx, [cls_logits[i] for i in order]


def forward_raw(model, raw):
    """in-memory raw → (d32, d16). 후보 선별용(빠른 경로)."""
    with torch.no_grad():
        x_int8, _ = preprocess(raw)
        d32, d16 = model(x_int8.squeeze(0))
    return d32, d16


# ────────────────────────────────────────────────────────────────────────────
# 한 챌린지 생성: pool 개 후보 중 target ROI 쏠림 최대 후보 선별 → 저장
# ────────────────────────────────────────────────────────────────────────────

def make_one_boundary_away(model, rois, target_idx, pool, score_mode,
                           bin_path, json_path, ch_idx):
    """pool 개 랜덤 입력을 뽑아 target ROI 쏠림 점수가 최대인 것을 선별.

    선별은 in-memory raw forward 로(빠름), 최종 저장 fingerprint 는 .bin 재로딩
    후 forward 로 기록해 round-trip safe 하게 만든다 (verifier 와 동일 경로).
    """
    target_roi = rois[target_idx]

    best_raw = None
    best_score = -float("inf")
    best_argmax = None
    best_sorted = None

    for _ in range(pool):
        raw = init_raw()                      # (1, 3, 360, 640) float [0,255]
        d32, d16 = forward_raw(model, raw)
        cls_l = roi_cls_logits(d32, d16, target_roi)
        s, amax, srt = skew_score(cls_l, score_mode)
        if s > best_score:
            best_score, best_raw = s, raw.detach().clone()
            best_argmax, best_sorted = amax, srt

    # 승자 저장 → 재로딩 → forward (저장된 logit 은 반드시 round-trip 값이어야 함)
    save_raw_bin(best_raw, bin_path)
    x_int8, _ = load_input(str(bin_path), preprocessed=False)
    with torch.no_grad():
        d32, d16 = model(x_int8.squeeze(0))

    logits = extract_roi_logits(d32, d16, rois)
    topk   = extract_roi_topk(d32, d16, rois)

    # 재로딩 후 target ROI 의 실제 쏠림(저장값 기준) 재계산 — 메타 기록용
    cls_reload = [logits[target_idx]["cls"][i]
                  for i in range(len(target_roi["cls_ch"]))]
    score_reload, argmax_reload, sorted_reload = skew_score(cls_reload, score_mode)
    probs = torch.softmax(torch.tensor(cls_reload), dim=-1).tolist()

    r = target_roi
    payload = {
        "challenge_idx": ch_idx,
        "mode": "boundary_away_random",
        "score_mode": score_mode,
        "pool_size": pool,
        # 선별 결과 메타 (분석용; verifier 는 무시 가능)
        "target_roi_idx": target_idx,
        "target_roi_cell": {
            "head": "S32" if r["head_idx"] == 0 else "S16",
            "anchor": r["anchor"], "gy": r["gy"], "gx": r["gx"],
        },
        "target_argmax_class": argmax_reload,
        "target_argmax_class_name": CLASS_NAMES[argmax_reload],
        "skew_score": score_reload,            # top1-top2 (margin) 또는 top1
        "skew_score_at_selection": best_score,  # 선별 시점(raw forward) 값
        "target_sorted_cls_logits": sorted_reload,
        "target_softmax": probs,
        # challenge 최적화 전용 필드: 의미 없음 → null (키는 보존)
        "converged": None,
        "iters_to_best": 0,
        # 핵심: boundary-aware 세트와 동일 ROI 에서의 logit/topk (verifier 호환)
        "rois_logits": logits,
        "rois_topk":   topk,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return score_reload, argmax_reload


# ────────────────────────────────────────────────────────────────────────────
# ROI 정의 확보 (boundary-aware fingerprint 우선 재사용)
# ────────────────────────────────────────────────────────────────────────────

def resolve_rois(model, fingerprint_path):
    if fingerprint_path.exists():
        with open(fingerprint_path, encoding="utf-8") as f:
            fp = json.load(f)
        print(f"[ROI] loaded {len(fp['rois'])} ROIs from {fingerprint_path}")
        return fp
    print(f"[ROI] {fingerprint_path} 없음 → select_golden_rois 로 새로 생성")
    rois = select_golden_rois(model, K=K_GOLDEN)
    return {"K": K_GOLDEN, "mode": "boundary_away_random",
            "rois": rois, "class_names": CLASS_NAMES,
            "note": "auto-generated (fingerprint 없을 때)"}


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--num",  type=int, default=100,
                   help="생성할 챌린지 수 (default 100)")
    p.add_argument("--pool", type=int, default=300,
                   help="챌린지당 랜덤 후보 수 (rejection sampling). "
                        "클수록 더 쏠린(경계에서 먼) 입력 선별. default 300")
    p.add_argument("--score", choices=["margin", "top1"], default="margin",
                   help="쏠림 점수: margin=top1-top2(경계거리, 기본), top1=top1 logit")
    p.add_argument("--out",  type=str, default=str(DEFAULT_SAVE_DIR),
                   help=f"출력 디렉터리 (default {DEFAULT_SAVE_DIR})")
    p.add_argument("--seed", type=int, default=12345,
                   help="RNG seed (default 12345). 같은 seed → 같은 후보열.")
    p.add_argument("--fingerprint", type=str, default=str(DEFAULT_FINGERPRINT),
                   help=f"ROI 정의 JSON (default {DEFAULT_FINGERPRINT}). "
                        "없으면 select_golden_rois 로 자동 생성.")
    p.add_argument("--variant", type=str, default="ste",
                   choices=["ste", "numpy", "diff"],
                   help="load_model variant (default 'ste' — challenge 와 동일)")
    args = p.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"output dir : {out_dir}")
    print(f"target N   : {args.num}  (pool/challenge={args.pool}, score={args.score})")
    print(f"seed       : {args.seed}")

    # seed 고정 (init_raw 가 np.random + torch.rand/randn 사용)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("loading model ...")
    model, model_type = load_model(args.variant)
    model.to(DEVICE)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    print(f"  model_type={model_type}, classes={CLASS_NAMES}")

    fp = resolve_rois(model, Path(args.fingerprint))
    rois = fp["rois"]
    K = len(rois)

    # 출력 디렉터리에 fingerprint 사본 저장 — self-contained (evaluate_challenges 호환)
    with open(out_dir / "fingerprint_rois.json", "w", encoding="utf-8") as f:
        json.dump(fp, f, indent=2)

    with open(out_dir / "boundary_away_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "kind": "boundary_away_random",
            "num": args.num, "pool": args.pool, "score": args.score,
            "seed": args.seed, "variant": args.variant,
            "fingerprint_src": str(args.fingerprint),
            "raw_resolution": [640, 360, 3], "raw_dtype": "uint8",
            "note": "rejection sampling: target ROI 의 class 벡터 쏠림(margin) 최대 후보 선별. "
                    "boundary-AWARE(uniform) 의 반대 baseline.",
        }, f, indent=2)

    # 이미 있는 파일 건너뛰기 (resume)
    done = set()
    for pth in out_dir.glob("challenge_*.bin"):
        stem = pth.stem.split("_", 1)[1]
        if stem.isdigit():
            done.add(int(stem))
    print(f"existing   : {len(done)} bin files\n")

    scores, argmaxes = [], []
    for ch_idx in range(1, args.num + 1):
        if ch_idx in done:
            continue
        target_idx = (ch_idx - 1) % K
        bin_path  = out_dir / f"challenge_{ch_idx:03d}.bin"
        json_path = out_dir / f"challenge_{ch_idx:03d}.json"
        s, amax = make_one_boundary_away(
            model, rois, target_idx, args.pool, args.score,
            bin_path, json_path, ch_idx)
        scores.append(s)
        argmaxes.append(amax)
        r = rois[target_idx]
        head_name = "S32" if r["head_idx"] == 0 else "S16"
        print(f"  [ch {ch_idx:03d}] ROI[{target_idx:02d}] {head_name} "
              f"cell=({r['gy']:2d},{r['gx']:2d})  "
              f"{args.score}={s:7.3f}  argmax={CLASS_NAMES[amax]:>10s}  "
              f"→ {bin_path.name}")

    if scores:
        arr = np.array(scores)
        print(f"\n[done] generated {len(scores)} challenges in {out_dir}")
        print(f"  skew({args.score})  mean={arr.mean():.3f}  median={np.median(arr):.3f}  "
              f"min={arr.min():.3f}  max={arr.max():.3f}")
        # argmax 클래스 분포 (자연 argmax 라 편중 확인용)
        uniq, cnt = np.unique(np.array(argmaxes), return_counts=True)
        dist = "  ".join(f"{CLASS_NAMES[u]}={c}" for u, c in zip(uniq, cnt))
        print(f"  argmax class dist : {dist}")
    else:
        print("\n[done] nothing to generate (all exist).")


if __name__ == "__main__":
    main()
