#!/usr/bin/env python3
"""
ResNet-20 챌린지 참조 Top-K 생성기 (verify-only 검증의 기준값).

용도:
    원본(골든) 가중치로 챌린지 PNG 들에 대한 Top-K 예측을 미리 뽑아
    JSON 으로 저장. 다른 PC 의 verify_resnet.py 가 이 JSON 과 비교해
    BFA 등 가중치 변조를 탐지.

기본 사용:
    # ./save/2025-09-09/resnet20_cifar10_best_weights.pth + ./data/.../random/*.png
    python generate_reference_topk.py

전체 인자 지정:
    python generate_reference_topk.py \
        --model resnet20_quan \
        --weights ./save/2025-09-09/resnet20_cifar10_best_weights.pth \
        --challenge_dir ./data/challenge_bench/resnet20_quan_w0/random \
        --output reference_topk_resnet20.json \
        --topk 5 \
        --device cpu

출력 JSON 형식:
    {
      "model": "resnet20_quan",
      "num_classes": 10,
      "image_size": [32, 32],
      "weights": "...",
      "challenge_dir": "...",
      "topk": 5,
      "device": "cpu",
      "torch_version": "2.x.x",
      "num_items": 100,
      "items": [
        {
          "filename": "challenge_image_1.png",
          "idx": 1,
          "topk_idx": [3, 5, 0, 7, 1],
          "topk_val": [0.92341234, 0.04123, ...]
        },
        ...
      ]
    }

⚠️  bit-exact 재현성:
    - device 는 cpu 권장 (GPU/CUDA 버전 차이로 마지막 비트 흔들림)
    - torch.use_deterministic_algorithms(True) 강제
    - 다른 PC 의 verify 스크립트도 동일 옵션 사용해야 함
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T


# ── bit-exact 재현성 강제 ────────────────────────────────────────────────────
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # CUDA 결정성 요구
torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(0)


# ── models/ 패키지 import (스크립트와 같은 폴더에 있다고 가정) ───────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from models.quan_resnet_cifar import resnet20_quan  # noqa: E402


# ────────────────────────────────────────────────────────────────────────────
# Model / weights
# ────────────────────────────────────────────────────────────────────────────

def build_model(name: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    if name == "resnet20_quan":
        model = resnet20_quan(num_classes=num_classes)
    else:
        raise ValueError(
            f"지원되지 않는 모델: {name} (현재 'resnet20_quan' 만 지원)"
        )
    return model.to(device).eval()


def load_weights(model: torch.nn.Module, weight_path: str, device: torch.device) -> torch.nn.Module:
    if not Path(weight_path).is_file():
        raise FileNotFoundError(f"가중치 파일 없음: {weight_path}")

    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(f"[ref] strict=False 로 로드 — missing={len(result.missing_keys)} "
              f"unexpected={len(result.unexpected_keys)}")
        if result.missing_keys:
            print(f"       first missing: {result.missing_keys[:3]}")
        if result.unexpected_keys:
            print(f"       first unexpected: {result.unexpected_keys[:3]}")
    return model


# ────────────────────────────────────────────────────────────────────────────
# Challenges
# ────────────────────────────────────────────────────────────────────────────

def load_challenges(challenge_dir: str, image_size: tuple) -> tuple:
    """challenge_image_*.png 들을 인덱스 순으로 로드해 (paths, tensor) 반환."""
    d = Path(challenge_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"챌린지 폴더 없음: {challenge_dir}")

    # 파일명에서 숫자 인덱스 추출해 정렬 (challenge_image_2.png < challenge_image_10.png)
    def idx_of(p: Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except ValueError:
            return 10**9  # 못 읽으면 뒤로

    paths = sorted(d.glob("challenge_image_*.png"), key=idx_of)
    if not paths:
        raise FileNotFoundError(
            f"챌린지 PNG 없음: {challenge_dir} (challenge_image_*.png 패턴 매칭 0)"
        )

    tfm = T.Compose([T.ToTensor()])  # PIL → [0,1] float32 tensor (C,H,W)
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if img.size != (image_size[1], image_size[0]):  # PIL.size 는 (W,H)
            raise ValueError(
                f"{p.name} 크기 불일치: 기대 {image_size} (H,W), 실제 {img.size[::-1]}"
            )
        imgs.append(tfm(img).unsqueeze(0))
    return paths, torch.cat(imgs, dim=0)


# ────────────────────────────────────────────────────────────────────────────
# Top-K
# ────────────────────────────────────────────────────────────────────────────

def compute_topk(
    model: torch.nn.Module,
    batch: torch.Tensor,
    k: int,
    device: torch.device,
    batch_size: int = 64,
) -> tuple:
    """배치 분할 추론으로 Top-K (확률, 인덱스) 추출."""
    all_idx, all_val = [], []
    with torch.inference_mode():
        for i in range(0, batch.size(0), batch_size):
            chunk = batch[i : i + batch_size].to(device)
            logits = model(chunk)
            probs = F.softmax(logits, dim=1)
            vals, idx = torch.topk(probs, k=k, dim=1)
            all_idx.append(idx.cpu())
            all_val.append(vals.cpu())
    return torch.cat(all_idx, dim=0), torch.cat(all_val, dim=0)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--model", default="resnet20_quan",
        help="모델 이름 (현재 'resnet20_quan' 만 지원)",
    )
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument(
        "--image_size", type=int, nargs=2, default=[32, 32],
        metavar=("H", "W"), help="입력 해상도 (H W)",
    )
    parser.add_argument(
        "--weights",
        default="./save/2025-09-09/resnet20_cifar10_best_weights.pth",
        help="골든(정상) 가중치 .pth 경로",
    )
    parser.add_argument(
        "--challenge_dir",
        default="./data/challenge_bench/resnet20_quan_w0/random",
        help="챌린지 PNG 폴더 (challenge_image_*.png)",
    )
    parser.add_argument(
        "--output", default="reference_topk_resnet20.json",
        help="저장할 참조 Top-K JSON 경로",
    )
    parser.add_argument("--topk", type=int, default=5, help="Top-K 의 K 값")
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu",
        help="추론 디바이스 (bit-exact 재현성 위해 cpu 권장)",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--verbose", action="store_true",
        help="per-image Top-K 인쇄",
    )
    args = parser.parse_args()

    # ── device 결정 ───────────────────────────────────────────────────────
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[ref] CUDA 미사용 환경 — cpu 로 전환")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"[ref] model       : {args.model}")
    print(f"[ref] num_classes : {args.num_classes}")
    print(f"[ref] image_size  : {tuple(args.image_size)}")
    print(f"[ref] device      : {device}")
    print(f"[ref] torch       : {torch.__version__}")
    print(f"[ref] weights     : {args.weights}")
    print(f"[ref] challenges  : {args.challenge_dir}")

    # ── 모델 + 가중치 ─────────────────────────────────────────────────────
    model = build_model(args.model, args.num_classes, device)
    model = load_weights(model, args.weights, device)

    # ── 챌린지 로드 ───────────────────────────────────────────────────────
    paths, batch = load_challenges(args.challenge_dir, tuple(args.image_size))
    print(f"[ref] loaded {len(paths)} challenges, batch shape={tuple(batch.shape)}")

    # ── Top-K 추론 ────────────────────────────────────────────────────────
    k = min(args.topk, args.num_classes)
    topk_idx, topk_val = compute_topk(model, batch, k, device, args.batch_size)

    # ── 결과 직렬화 ───────────────────────────────────────────────────────
    items = []
    for path, idx_row, val_row in zip(paths, topk_idx.tolist(), topk_val.tolist()):
        try:
            file_idx = int(path.stem.split("_")[-1])
        except ValueError:
            file_idx = -1
        items.append({
            "filename": path.name,
            "idx": file_idx,
            "topk_idx": idx_row,
            "topk_val": [round(v, 8) for v in val_row],
        })

    out = {
        "model": args.model,
        "num_classes": args.num_classes,
        "image_size": list(args.image_size),
        "weights": str(Path(args.weights).resolve()),
        "challenge_dir": str(Path(args.challenge_dir).resolve()),
        "topk": k,
        "device": str(device),
        "torch_version": torch.__version__,
        "num_items": len(items),
        "items": items,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[ref] saved → {args.output}  ({len(items)} items, k={k})")

    if args.verbose:
        print()
        for it in items:
            print(f"  {it['filename']:<32} top1={it['topk_idx'][0]:>2d} "
                  f"({it['topk_val'][0]:.6f})  topK={it['topk_idx']}")


if __name__ == "__main__":
    main()
