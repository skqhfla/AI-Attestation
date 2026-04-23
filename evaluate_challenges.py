#!/usr/bin/env python3
"""
Wyze 챌린지 평가 스크립트 (BFA 전/후 비교용)
================================================

용도:
  1. Clean 모델에서 100개 .bin 을 돌려 baseline 기록
  2. BFA 공격 후 변조된 .pth 를 로드해 동일 .bin 을 다시 돌림
  3. 두 결과를 diff 하여 attestation max_dev / detection confidence 변화 확인

사용:
    # Baseline (pre-BFA)
    python evaluate_challenges.py --output pre_bfa.json

    # Post-BFA (BFA 로 저장된 .pth 로드)
    python evaluate_challenges.py \\
        --weights results/wyze/weights/wyze_bfa_bit30_conf0.xxxx.pth \\
        --output post_bfa.json

    # 비교 (두 JSON 차분)
    python evaluate_challenges.py --diff pre_bfa.json post_bfa.json

출력 지표:
  att_max_dev: 저장된 per-challenge fingerprint 와 현재 forward 결과의 logit 최대 편차
               → BFA 비트 플립 시 급증 예상 (attestation signal)
  num_detections, max_conf: 일반 YOLO detection 관점의 변화
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

from classify import load_model, load_input, decode, nms, CLASS_NAMES

DEVICE = torch.device('cpu')
DEFAULT_SAVE_DIR = SCRIPT_DIR / "data" / "challenge" / "wyze_ste_multiroi"
ATTEST_THRESHOLD = 0.5     # att_max_dev 이하면 PASS


# ────────────────────────────────────────────────────────────────────────────
# Helpers (generate_challenge_wyze.py 와 동일 로직)
# ────────────────────────────────────────────────────────────────────────────

def extract_roi_logits(d32, d16, rois):
    heads = [d32, d16]
    out = []
    for r in rois:
        head = heads[r['head_idx']]
        gy, gx = r['gy'], r['gx']
        out.append({
            'obj': float(head[r['obj_ch'], gy, gx].detach()),
            'cls': [float(head[c, gy, gx].detach()) for c in r['cls_ch']],
        })
    return out


def max_dev_vs_stored(current, stored):
    md = 0.0
    for c, s in zip(current, stored):
        md = max(md, abs(c['obj'] - s['obj']))
        for cc, ss in zip(c['cls'], s['cls']):
            md = max(md, abs(cc - ss))
    return md


# ────────────────────────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────────────────────────

def evaluate(model, rois, save_dir, verbose=False):
    """모든 .bin 을 돌려 per-challenge 결과 반환."""
    results = []
    paths = sorted(save_dir.glob("challenge_*.bin"))
    for bin_path in paths:
        json_path = bin_path.with_suffix('.json')
        if not json_path.exists():
            if verbose:
                print(f"  [SKIP] {bin_path.name}: no .json")
            continue
        with open(json_path) as f:
            stored_fp = json.load(f)['rois_logits']

        x_int8, _ = load_input(str(bin_path), preprocessed=False)
        with torch.no_grad():
            d32, d16 = model(x_int8.squeeze(0))

        # Attestation fingerprint deviation
        current = extract_roi_logits(d32, d16, rois)
        att = max_dev_vs_stored(current, stored_fp)

        # Classical YOLO detection
        d32_np = d32.cpu().numpy()
        d16_np = d16.cpu().numpy()
        dets = decode(d32_np, d16_np, conf_thresh=0.25)
        dets = nms(dets, iou_thresh=0.45)

        cls_counts = [0] * 5
        for d in dets:
            cls_counts[d['class_id']] += 1

        results.append({
            'bin': bin_path.name,
            'att_max_dev': float(att),
            'num_detections': len(dets),
            'max_conf': float(max((d['confidence'] for d in dets), default=0.0)),
            'class_counts': cls_counts,
        })

        if verbose:
            tag = "OK  " if att < ATTEST_THRESHOLD else "FAIL"
            print(f"  [{tag}] {bin_path.name}  att_max_dev={att:.4f}  "
                  f"dets={len(dets):2d}  max_conf={results[-1]['max_conf']:.4f}")
    return results


def summarize(results, label=""):
    if not results:
        print(f"{label}: no results"); return
    ad = np.array([r['att_max_dev']     for r in results])
    nd = np.array([r['num_detections']  for r in results])
    mc = np.array([r['max_conf']        for r in results])
    cc = np.array([r['class_counts']    for r in results]).sum(axis=0)
    passed = int((ad < ATTEST_THRESHOLD).sum())

    print(f"\n{'='*64}")
    print(f"{label or 'Results'}")
    print(f"{'='*64}")
    print(f"Challenges evaluated: {len(results)}")
    print(f"\nATTESTATION (stored fingerprint 대비 현재 forward 편차):")
    print(f"  passed (<{ATTEST_THRESHOLD}): {passed}/{len(results)}")
    print(f"  att_max_dev   mean={ad.mean():.4f}   max={ad.max():.4f}   min={ad.min():.4f}")
    print(f"\nDETECTION (decode + NMS):")
    print(f"  #detections/challenge  mean={nd.mean():.2f}  max={nd.max()}  min={nd.min()}")
    print(f"  max_confidence         mean={mc.mean():.4f}  max={mc.max():.4f}")
    print(f"  class distribution (total detections):")
    for i, c in enumerate(cc):
        print(f"    {CLASS_NAMES[i]:10s}: {c}")


# ────────────────────────────────────────────────────────────────────────────
# Diff mode
# ────────────────────────────────────────────────────────────────────────────

def diff_mode(pre_path, post_path):
    with open(pre_path)  as f: pre  = json.load(f)
    with open(post_path) as f: post = json.load(f)
    pre_map  = {r['bin']: r for r in pre['results']}
    post_map = {r['bin']: r for r in post['results']}

    print(f"\n{'='*96}")
    print(f"DIFF: pre={pre['weights']}  →  post={post['weights']}")
    print(f"{'='*96}")
    print(f"{'challenge':<25} {'pre_att':>9} {'post_att':>9} {'Δatt':>10} "
          f"{'pre_conf':>9} {'post_conf':>9} {'Δdets':>7}")
    print(f"{'-'*96}")

    rows = []
    for name in sorted(pre_map.keys() & post_map.keys()):
        p, q = pre_map[name], post_map[name]
        dad = q['att_max_dev'] - p['att_max_dev']
        ddt = q['num_detections'] - p['num_detections']
        flag = " !!" if dad > 0.1 else ""
        print(f"{name:<25} {p['att_max_dev']:>9.4f} {q['att_max_dev']:>9.4f} "
              f"{dad:>+10.4f} {p['max_conf']:>9.4f} {q['max_conf']:>9.4f} "
              f"{ddt:>+7}{flag}")
        rows.append((dad, ddt, p['max_conf'] - q['max_conf']))

    da   = np.array([r[0] for r in rows])
    dnd  = np.array([r[1] for r in rows])
    dmc  = np.array([r[2] for r in rows])
    detected = int((da > 0.1).sum())

    print(f"\n{'─'*64}")
    print(f"ATTESTATION deviation (post − pre):")
    print(f"  mean={da.mean():+.4f}   max={da.max():+.4f}   min={da.min():+.4f}")
    print(f"  attested (Δatt > 0.1): {detected}/{len(rows)}")
    print(f"DETECTION change (post − pre):")
    print(f"  Δ#detections   mean={dnd.mean():+.2f}  max={dnd.max():+d}  min={dnd.min():+d}")
    print(f"  Δmax_conf (pre − post)  mean={dmc.mean():+.4f}")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument('--save_dir', default=str(DEFAULT_SAVE_DIR),
                        help="challenge_*.bin / .json 디렉토리")
    parser.add_argument('--weights', default=None,
                        help="선택: 모델 state_dict(.pth) 경로 (BFA 후 사용)")
    parser.add_argument('--output',  default=None,
                        help="결과를 JSON 으로 저장 (diff 에 사용)")
    parser.add_argument('--diff', nargs=2, metavar=('PRE', 'POST'),
                        help="모델 실행 대신 두 결과 JSON 을 비교")
    parser.add_argument('--verbose', action='store_true',
                        help="per-challenge 상세 출력")
    args = parser.parse_args()

    if args.diff:
        diff_mode(args.diff[0], args.diff[1])
        return

    save_dir = Path(args.save_dir)

    print(f"Device: {DEVICE}")
    print("Loading model: classify.load_model('ste')")
    model, _ = load_model('ste')
    model.to(DEVICE)
    model.eval()

    if args.weights:
        print(f"Loading weights override: {args.weights}")
        sd = torch.load(args.weights, map_location=DEVICE)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        result = model.load_state_dict(sd, strict=False)
        if result.missing_keys or result.unexpected_keys:
            print(f"  strict=False: missing={len(result.missing_keys)} "
                  f"unexpected={len(result.unexpected_keys)}")
            if result.missing_keys:
                print(f"    first missing: {result.missing_keys[:3]}")
            if result.unexpected_keys:
                print(f"    first unexpected: {result.unexpected_keys[:3]}")
        print("  loaded.")
    else:
        print("(default weights — BFA 이전 baseline)")

    with open(save_dir / "fingerprint_rois.json") as f:
        rois = json.load(f)['rois']
    print(f"Golden ROIs: {len(rois)}")

    print(f"\nEvaluating challenges in {save_dir} ...")
    results = evaluate(model, rois, save_dir, verbose=args.verbose)

    label = f"weights={args.weights}" if args.weights else "default weights (baseline)"
    summarize(results, label=label)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'weights': args.weights or 'default',
                'save_dir': str(save_dir),
                'attest_threshold': ATTEST_THRESHOLD,
                'results': results,
            }, f, indent=2)
        print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
