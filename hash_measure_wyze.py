#!/usr/bin/env python3
"""
hash_measure_wyze.py — Wyze EdgeAI 5-Class YOLO 용
================================================
hash_measure.py(CIFAR/ResNet/EfficientNet 용)와 동일 메트릭을
Wyze 모델 환경에 맞춰 재구성.

CIFAR 판과의 핵심 차이
----------------------
1. **가중치 표현이 두 가지**
     - weight_blob (device 가 실제로 쓰는 raw INT8 바이너리)  ← BFA 가 손대는 그 파일
     - model_blob  (layer topology / param 정의)
     - state_dict  (PyTorch 메모리상 텐서)
   세 가지를 모두 측정해 둔다. 무결성 검증 관점에선 weight_blob 해시가 본질.
2. **양자화 변환 단계가 불필요**: WyzeClassifySTE 가 이미 INT8 STE 양자화 모델.
   (CIFAR 판의 quantize(model) 호출 없음.)
3. **입력 텐서 shape**: 모델 직접 입력은 (3, 256, 448), 디바이스 raw 는
   (3, 360, 640) → preprocess 거쳐 (3, 256, 448) 로 들어감.
   둘 다 측정 (model-only / end-to-end).

사용
----
    python hash_measure_wyze.py                                # 기본 (N_HASH=1000, N_INFER=100)
    python hash_measure_wyze.py --n-hash 200 --n-infer 50      # 빠르게
    python hash_measure_wyze.py --variant numpy                # numpy 모델 (backward 불가)
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
WYZE_DIR   = SCRIPT_DIR / "wyze_model"
sys.path.insert(0, str(WYZE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

# Wyze 모듈
from classify import load_model, preprocess, CAM_H, CAM_W                # noqa: E402
from model_torch_ste import MODEL_BLOB_PATH, WEIGHT_BLOB_PATH             # noqa: E402


# ────────────────────────────────────────────────────────────────────────────
# 해시 측정
# ────────────────────────────────────────────────────────────────────────────

def _stats(times_ms):
    return {
        "trials":  len(times_ms),
        "avg_ms":  statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms":  min(times_ms),
        "max_ms":  max(times_ms),
        "stdev_ms": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
    }


def hash_file(path: Path, n_trials: int):
    """단일 바이너리 파일을 1회 hash. 같은 입력에 대한 SHA256 계산을
    n_trials 회 반복해 시간 분포만 수집한다.

    원본 hash_measure.py 와 똑같이: 같은 데이터를 매 trial 마다 새로 해싱
    (캐시/메모리 effect 흡수)."""
    data = path.read_bytes()
    times = []
    digest = None
    for _ in range(n_trials):
        h = hashlib.sha256()
        t0 = time.perf_counter()
        h.update(data)
        d  = h.hexdigest()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
        digest = d  # 매 trial 같은 값
    return {
        "path":          str(path),
        "size_bytes":    len(data),
        "size_MB":       len(data) / (1024 * 1024),
        "sha256":        digest,
        **_stats(times),
    }


def hash_state_dict(model: torch.nn.Module, n_trials: int):
    """원본 hash_measure.py 의 'state_dict 순회 + bytes 누적' 방식 그대로.

    매 trial 마다 sha256 누적 객체를 새로 만들고 모든 텐서를 직렬화한 뒤
    update → hexdigest 까지의 시간을 잰다."""
    sd = model.state_dict()
    sizes = 0
    times = []
    digest = None
    for trial in range(n_trials):
        sha = hashlib.sha256()
        t0 = time.perf_counter()
        size_this = 0
        for k, v in sd.items():
            if isinstance(v, torch.Tensor):
                b = v.detach().cpu().numpy().tobytes()
                sha.update(b)
                size_this += len(b)
        d = sha.hexdigest()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
        if trial == 0:
            sizes = size_this
            digest = d
    return {
        "n_tensors":     sum(1 for v in sd.values() if isinstance(v, torch.Tensor)),
        "size_bytes":    sizes,
        "size_MB":       sizes / (1024 * 1024),
        "sha256":        digest,
        **_stats(times),
    }


# ────────────────────────────────────────────────────────────────────────────
# 추론 시간 측정
# ────────────────────────────────────────────────────────────────────────────

def inference_time_model_only(model, model_type: str, n_trials: int, n_warmup: int = 10):
    """모델 forward 만의 시간. 입력 shape = (3, 256, 448), 모델 입력 단계 기준."""
    # WyzeClassifySTE 는 (3, 256, 448) 직접 받음 (classify.py 의 squeeze(0) 동작 그대로)
    if model_type == "numpy":
        dummy = np.random.randint(-128, 128, size=(3, 256, 448), dtype=np.int8)
        # warmup
        for _ in range(n_warmup):
            _ = model.forward(dummy)
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            _ = model.forward(dummy)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    else:
        dummy = torch.randn(3, 256, 448)
        # warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(dummy)
        times = []
        with torch.no_grad():
            for _ in range(n_trials):
                t0 = time.perf_counter()
                _ = model(dummy)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)
    return _stats(times)


def inference_time_end_to_end(model, model_type: str, n_trials: int, n_warmup: int = 5):
    """raw (360, 640, 3) uint8 → preprocess → model forward 까지의 wall-clock.

    device 가 실제로 거치는 전체 파이프라인이다. preprocess 가 stb_resize
    (ctypes/.so) 와 sRGB 변환을 포함하므로 model-only 시간보다 의미가 크다."""
    raw = np.random.randint(0, 256, size=(CAM_H, CAM_W, 3), dtype=np.uint8)

    # warmup
    for _ in range(n_warmup):
        x, _ = preprocess(raw)
        if model_type == "numpy":
            _ = model.forward(x.squeeze(0).detach().numpy().astype(np.int8))
        else:
            with torch.no_grad():
                _ = model(x.squeeze(0))

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        x, _ = preprocess(raw)
        if model_type == "numpy":
            _ = model.forward(x.squeeze(0).detach().numpy().astype(np.int8))
        else:
            with torch.no_grad():
                _ = model(x.squeeze(0))
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return _stats(times)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def _print_section(title):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _print_kv(d, indent=2):
    pad = " " * indent
    for k, v in d.items():
        if isinstance(v, float):
            print(f"{pad}{k:<18}: {v:.4f}")
        else:
            print(f"{pad}{k:<18}: {v}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--variant", default="ste", choices=["ste", "numpy", "diff"],
                   help="모델 variant (default ste — challenge 와 동일)")
    p.add_argument("--n-hash",  type=int, default=1000,
                   help="해시 trial 수 (default 1000 — hash_measure.py 와 동일)")
    p.add_argument("--n-infer", type=int, default=100,
                   help="추론 trial 수 (default 100). model 이 무거우니 보수적으로.")
    p.add_argument("--n-warmup", type=int, default=10,
                   help="추론 warmup trial 수 (default 10)")
    p.add_argument("--weight-blob", default=str(WEIGHT_BLOB_PATH),
                   help="weight_blob 경로 (default wyze_model/weight_blob)")
    p.add_argument("--model-blob",  default=str(MODEL_BLOB_PATH),
                   help="model_blob 경로 (default wyze_model/model_blob)")
    p.add_argument("--out-json", default=None,
                   help="결과를 JSON 으로 dump 할 경로 (선택)")
    args = p.parse_args()

    print(f"variant = {args.variant}")
    print(f"hash trials  = {args.n_hash}")
    print(f"infer trials = {args.n_infer} (+ warmup {args.n_warmup})")

    # 모델 로딩
    print(f"\nloading model ...")
    t0 = time.perf_counter()
    model, model_type = load_model(args.variant)
    if model_type == "torch":
        model.eval()
        for prm in model.parameters():
            prm.requires_grad_(False)
    t1 = time.perf_counter()
    print(f"  model loaded in {(t1-t0)*1000:.1f} ms  (type={model_type})")

    results = {
        "variant":   args.variant,
        "model_type": model_type,
        "host": {
            "platform": sys.platform,
            "python":   sys.version.split()[0],
            "torch":    torch.__version__,
        },
        "config": {
            "n_hash":  args.n_hash,
            "n_infer": args.n_infer,
            "n_warmup": args.n_warmup,
        },
    }

    # 1) weight_blob 해시
    _print_section("Hash: weight_blob  (device-side raw INT8 weights)")
    r = hash_file(Path(args.weight_blob), args.n_hash)
    _print_kv(r)
    results["weight_blob"] = r

    # 2) model_blob 해시 (topology)
    _print_section("Hash: model_blob  (layer topology / param defs)")
    r = hash_file(Path(args.model_blob), args.n_hash)
    _print_kv(r)
    results["model_blob"] = r

    # 3) state_dict 해시 (PyTorch 메모리)
    if model_type == "torch":
        _print_section("Hash: model.state_dict()  (PyTorch in-memory)")
        r = hash_state_dict(model, args.n_hash)
        _print_kv(r)
        results["state_dict"] = r
    else:
        print("\n[skip] state_dict 해시는 torch 모델에만 해당 (variant=numpy 는 생략)")

    # 4) 추론 시간 — model only
    _print_section("Inference time: model only  (input shape = 3 x 256 x 448)")
    r = inference_time_model_only(model, model_type, args.n_infer, args.n_warmup)
    _print_kv(r)
    results["inference_model_only_ms"] = r

    # 5) 추론 시간 — end-to-end (preprocess 포함)
    _print_section("Inference time: end-to-end  (640x360x3 raw → preprocess → model)")
    r = inference_time_end_to_end(model, model_type, args.n_infer, args.n_warmup)
    _print_kv(r)
    results["inference_e2e_ms"] = r

    # 요약 한 줄
    _print_section("Summary")
    print(f"  weight_blob sha256  : {results['weight_blob']['sha256']}")
    print(f"  weight_blob size    : {results['weight_blob']['size_MB']:.4f} MB")
    print(f"  weight_blob hash    : {results['weight_blob']['avg_ms']:.4f} ms (avg of {args.n_hash})")
    if "state_dict" in results:
        print(f"  state_dict sha256   : {results['state_dict']['sha256']}")
        print(f"  state_dict hash     : {results['state_dict']['avg_ms']:.4f} ms (avg of {args.n_hash})")
    print(f"  inference model-only: {results['inference_model_only_ms']['avg_ms']:.3f} ms")
    print(f"  inference e2e       : {results['inference_e2e_ms']['avg_ms']:.3f} ms")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
