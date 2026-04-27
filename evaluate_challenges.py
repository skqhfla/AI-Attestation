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


def extract_roi_topk(d32, d16, rois):
    """BAIV-style: 각 ROI 의 5 cls 채널을 내림차순 정렬한 class index list."""
    heads = [d32, d16]
    out = []
    for r in rois:
        head = heads[r['head_idx']]
        gy, gx = r['gy'], r['gx']
        scores = [(float(head[c, gy, gx].detach()), i) for i, c in enumerate(r['cls_ch'])]
        scores.sort(key=lambda x: (-x[0], x[1]))
        out.append([i for _, i in scores])
    return out


def topk_from_logits(stored_fp):
    """옛 fingerprint (.json 에 'rois_topk' 가 없는 경우) 호환:
    stored_fp = [{'obj':..., 'cls':[5 floats]}, ...] → ROI 마다 length-5 ranking.
    """
    out = []
    for s in stored_fp:
        scores = [(s['cls'][i], i) for i in range(len(s['cls']))]
        scores.sort(key=lambda x: (-x[0], x[1]))
        out.append([i for _, i in scores])
    return out


def topk_mismatch(current_topk, stored_topk, k):
    """K 위까지의 prefix 가 다른 ROI 의 개수 (0 ~ len(rois))."""
    return sum(1 for cur, sto in zip(current_topk, stored_topk)
               if cur[:k] != sto[:k])


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
    """모든 .bin 을 돌려 per-challenge 결과 반환.

    저장 지표:
      - att_max_dev   : raw logit 편차 (기존, logit 노출 가정 위협 모델)
      - topk{1,2,3}_mis: BAIV-style Top-K class index mismatch (ROI 단위)
      - any_topk_mis  : k=5 prefix (=전체 5 위 순서) 불일치 ROI 수
    """
    results = []
    paths = sorted(save_dir.glob("challenge_*.bin"))
    for bin_path in paths:
        json_path = bin_path.with_suffix('.json')
        if not json_path.exists():
            if verbose:
                print(f"  [SKIP] {bin_path.name}: no .json")
            continue
        with open(json_path) as f:
            stored = json.load(f)
        stored_fp = stored['rois_logits']
        stored_topk = stored.get('rois_topk') or topk_from_logits(stored_fp)

        x_int8, _ = load_input(str(bin_path), preprocessed=False)
        with torch.no_grad():
            d32, d16 = model(x_int8.squeeze(0))

        # Attestation fingerprint deviation (raw logit 기반, 기존 지표)
        current = extract_roi_logits(d32, d16, rois)
        att = max_dev_vs_stored(current, stored_fp)

        # BAIV-style Top-K matching (per-ROI class index ordering)
        current_topk = extract_roi_topk(d32, d16, rois)
        mis_k1 = topk_mismatch(current_topk, stored_topk, 1)
        mis_k2 = topk_mismatch(current_topk, stored_topk, 2)
        mis_k3 = topk_mismatch(current_topk, stored_topk, 3)
        any_mis = topk_mismatch(current_topk, stored_topk, 5)  # 5 = 전체 순서

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
            'topk1_mis':   int(mis_k1),
            'topk2_mis':   int(mis_k2),
            'topk3_mis':   int(mis_k3),
            'any_topk_mis': int(any_mis),
            'num_rois':    len(rois),
            'num_detections': len(dets),
            'max_conf': float(max((d['confidence'] for d in dets), default=0.0)),
            'class_counts': cls_counts,
        })

        if verbose:
            tag_a = "OK" if att       < ATTEST_THRESHOLD else "FAIL"
            tag_t = "OK" if mis_k1 == 0                  else "FAIL"
            print(f"  [att:{tag_a:4s} top1:{tag_t:4s}] {bin_path.name}  "
                  f"att={att:.4f}  top1_mis={mis_k1:2d}/{len(rois)}  "
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

    # Top-K (BAIV-style)
    t1 = np.array([r.get('topk1_mis', 0)    for r in results])
    t2 = np.array([r.get('topk2_mis', 0)    for r in results])
    t3 = np.array([r.get('topk3_mis', 0)    for r in results])
    ta = np.array([r.get('any_topk_mis', 0) for r in results])
    n_rois = results[0].get('num_rois', 16) if results else 16
    det_k1 = int((t1 > 0).sum())
    det_k2 = int((t2 > 0).sum())
    det_k3 = int((t3 > 0).sum())
    det_any = int((ta > 0).sum())

    print(f"\n{'='*64}")
    print(f"{label or 'Results'}")
    print(f"{'='*64}")
    print(f"Challenges evaluated: {len(results)}")
    print(f"\nATTESTATION (raw logit 편차, stored fingerprint 대비):")
    print(f"  passed (<{ATTEST_THRESHOLD}): {passed}/{len(results)}")
    print(f"  att_max_dev   mean={ad.mean():.4f}   max={ad.max():.4f}   min={ad.min():.4f}")
    print(f"\nTOP-K MATCHING (BAIV-style, per-ROI class index ordering):")
    print(f"  detected (any-ROI Top-1 mismatch): {det_k1}/{len(results)}  "
          f"(평균 mismatched ROIs = {t1.mean():.2f}/{n_rois})")
    print(f"  detected (any-ROI Top-2 mismatch): {det_k2}/{len(results)}  "
          f"(평균 {t2.mean():.2f}/{n_rois})")
    print(f"  detected (any-ROI Top-3 mismatch): {det_k3}/{len(results)}  "
          f"(평균 {t3.mean():.2f}/{n_rois})")
    print(f"  detected (any-ROI full ordering mismatch): {det_any}/{len(results)}  "
          f"(평균 {ta.mean():.2f}/{n_rois})")
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

    print(f"\n{'='*108}")
    print(f"DIFF: pre={pre['weights']}  →  post={post['weights']}")
    print(f"{'='*108}")
    print(f"{'challenge':<25} {'pre_att':>9} {'post_att':>9} {'Δatt':>10} "
          f"{'top1':>5} {'top3':>5} {'pre_conf':>9} {'post_conf':>9} {'Δdets':>7}")
    print(f"{'-'*108}")

    rows = []
    for name in sorted(pre_map.keys() & post_map.keys()):
        p, q = pre_map[name], post_map[name]
        dad = q['att_max_dev'] - p['att_max_dev']
        ddt = q['num_detections'] - p['num_detections']
        t1  = q.get('topk1_mis', 0)
        t3  = q.get('topk3_mis', 0)
        flag = " !!" if (dad > 0.1 or t1 > 0) else ""
        print(f"{name:<25} {p['att_max_dev']:>9.4f} {q['att_max_dev']:>9.4f} "
              f"{dad:>+10.4f} {t1:>5d} {t3:>5d} "
              f"{p['max_conf']:>9.4f} {q['max_conf']:>9.4f} "
              f"{ddt:>+7}{flag}")
        rows.append((dad, ddt, p['max_conf'] - q['max_conf'], t1, t3))

    da   = np.array([r[0] for r in rows])
    dnd  = np.array([r[1] for r in rows])
    dmc  = np.array([r[2] for r in rows])
    pt1  = np.array([r[3] for r in rows])
    pt3  = np.array([r[4] for r in rows])
    detected_att  = int((da > 0.1).sum())
    detected_top1 = int((pt1 > 0).sum())
    detected_top3 = int((pt3 > 0).sum())

    print(f"\n{'─'*72}")
    print(f"ATTESTATION (raw logit 기반):")
    print(f"  Δatt   mean={da.mean():+.4f}   max={da.max():+.4f}   min={da.min():+.4f}")
    print(f"  attested (Δatt > 0.1): {detected_att}/{len(rows)}")
    print(f"\nTOP-K MATCHING (BAIV-style, post 의 ROI mismatch 수):")
    print(f"  detected (Top-1 any mismatch): {detected_top1}/{len(rows)}  "
          f"(post 평균 {pt1.mean():.2f} ROI mismatch)")
    print(f"  detected (Top-3 any mismatch): {detected_top3}/{len(rows)}  "
          f"(post 평균 {pt3.mean():.2f} ROI mismatch)")
    print(f"\nDETECTION change (post − pre):")
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
