#!/usr/bin/env python3
"""
Wyze Cam Pan V3 EdgeAI — 5-Class YOLO Detection Pipeline
=========================================================
Model selection + Preprocessing + Inference + Postprocessing.

Models:
  ste      — PyTorch STE (model_torch_ste.py, bit-exact + 미분가능, 기본값)
  numpy    — 원본 INT8 numpy 모델 (model.py, bit-exact, backward 불가)
  diff     — PyTorch float32 미분가능 근사 (model_torch_diff.py)

Usage:
    python classify.py frame.bin                           # STE 모델, raw 640x360
    python classify.py frame.bin --model numpy             # numpy 모델
    python classify.py --preprocessed dump.bin             # 전처리 완료 256x448x3
    python classify.py --preprocessed dump.bin --no-nms    # NMS 없이 decode까지만
    python classify.py frame.bin --conf 0.25 --iou 0.45
"""

import numpy as np
import sys
import os
import argparse

MODEL_H = 256
MODEL_W = 448

ANCHORS_S32 = [(93.0, 57.0), (103.0, 141.0), (209.0, 168.0)]
ANCHORS_S16 = [(16.0, 16.0), (42.0, 32.0), (43.0, 87.0)]
NUM_CLASSES = 5
NUM_ANCHORS = 3
CLASS_NAMES = ["person", "vehicle", "pet", "package", "face"]
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

CAM_W = 640
CAM_H = 360


# ═══════════════════════════════════════════════════════════════════════════
# Model Loader
# ═══════════════════════════════════════════════════════════════════════════

def load_model(variant='ste'):
    """Load model by variant name.

    Args:
        variant: 'numpy', 'diff', 'ste'

    Returns:
        model object, model_type ('numpy' or 'torch')
    """
    if variant == 'numpy':
        from model import WyzeClassify
        return WyzeClassify.load(), 'numpy'

    elif variant == 'diff':
        from model_torch_diff import WyzeClassifyDiff
        model = WyzeClassifyDiff()
        model.eval()
        return model, 'torch'

    elif variant == 'ste':
        from model_torch_ste import WyzeClassifySTE
        model = WyzeClassifySTE()
        model.eval()
        return model, 'torch'

    else:
        raise ValueError(f"Unknown variant: {variant}. Choose: numpy, diff, ste")


def run_inference(model, model_type, input_data):
    """Run model forward.

    Args:
        model: loaded model
        model_type: 'numpy' or 'torch'
        input_data: torch tensor (1, 3, 256, 448) from preprocess()
                    또는 int8 numpy array [3, 256, 448] (legacy)

    Returns:
        d32, d16 as numpy float32 arrays
    """
    import torch

    if model_type == 'numpy':
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.squeeze(0).detach().numpy().astype(np.int8)
        d32, d16 = model.forward(input_data)
    else:
        if isinstance(input_data, np.ndarray):
            x = torch.from_numpy(input_data.astype(np.float32))
        else:
            x = input_data.squeeze(0) if input_data.dim() == 4 else input_data
        with torch.no_grad():
            d32, d16 = model(x)
        d32, d16 = d32.numpy(), d16.numpy()

    return d32, d16


# ═══════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def _load_stb_resize():
    """Load libstb_resize.so (stb_image_resize BOX + sRGB, 디바이스 동일)."""
    import ctypes
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libstb_resize.so')
    lib = ctypes.CDLL(lib_path)
    lib.stb_resize_srgb_box.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.stb_resize_srgb_box.restype = None
    return lib

_stb_lib = None

def _stb_resize(src_hwc_u8, dst_h, dst_w):
    global _stb_lib
    if _stb_lib is None:
        _stb_lib = _load_stb_resize()
    src = np.ascontiguousarray(src_hwc_u8)
    dst = np.zeros((dst_h, dst_w, src.shape[2]), dtype=np.uint8)
    _stb_lib.stb_resize_srgb_box(src.ctypes.data, src.shape[1], src.shape[0],
                                  dst.ctypes.data, dst_w, dst_h, src.shape[2])
    return dst


def _preprocess_exact(raw_rgb):
    """stb bit-exact resize (내부용)."""
    resized = _stb_resize(raw_rgb, MODEL_H, MODEL_W)
    input_i8 = (resized.astype(np.int16) - 128).astype(np.int8)
    return input_i8.transpose(2, 0, 1)


def preprocess(raw_rgb):
    """Convert raw camera frame (640x360 RGB) to model input.

    STE 방식: forward는 디바이스와 100% bit-exact (stb_image_resize),
    backward는 미분 가능 근사 (sRGB + bilinear)로 gradient 전파.

    Args:
        raw_rgb: numpy array (360, 640, 3) uint8 RGB
                 또는 torch tensor (1, 3, 360, 640) float32 [0~255]

    Returns:
        torch tensor (1, 3, 256, 448) float32 — gradient 전파 가능
        resize_info: dict for coordinate mapping
    """
    import torch
    import torch.nn.functional as F

    if isinstance(raw_rgb, np.ndarray):
        x = torch.from_numpy(raw_rgb.astype(np.float32))
        x = x.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 360, 640]
    else:
        x = raw_rgb
        if x.dim() == 3:
            x = x.unsqueeze(0)

    _, _, orig_h, orig_w = x.shape

    # --- 미분 가능 경로 (backward용, sRGB + bilinear 근사) ---
    x_norm = x / 255.0
    linear = torch.where(x_norm <= 0.04045,
                         x_norm / 12.92,
                         ((x_norm + 0.055) / 1.055) ** 2.4)
    resized = F.interpolate(linear, size=(MODEL_H, MODEL_W),
                            mode='bilinear', align_corners=False)
    srgb = torch.where(resized <= 0.0031308,
                       resized * 12.92,
                       1.055 * resized.clamp(min=0).pow(1.0 / 2.4) - 0.055)
    approx = srgb * 255.0 - 128.0

    # --- bit-exact 경로 (forward용) ---
    raw_np = x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    raw_u8 = np.clip(np.round(raw_np), 0, 255).astype(np.uint8)
    exact_i8 = _preprocess_exact(raw_u8)
    exact = torch.from_numpy(exact_i8.astype(np.float32)).unsqueeze(0)

    # --- STE: forward=exact, backward=approx의 gradient ---
    output = (exact - approx).detach() + approx

    return output, {
        'orig_h': orig_h, 'orig_w': orig_w,
        'scale_x': MODEL_W / orig_w, 'scale_y': MODEL_H / orig_h,
    }


def load_input(path, preprocessed=False):
    """Load input file.

    Args:
        path: file path
        preprocessed: if True, load as 256x448x3 int8 (no preprocess)

    Returns:
        torch tensor (1, 3, 256, 448) float32, resize_info or None
    """
    import torch

    if preprocessed:
        data = np.fromfile(path, dtype=np.int8)
        assert data.size == MODEL_H * MODEL_W * 3, \
            f"Expected {MODEL_H * MODEL_W * 3} bytes, got {data.size}"
        chw = data.reshape(MODEL_H, MODEL_W, 3).transpose(2, 0, 1)
        return torch.from_numpy(chw.astype(np.float32)).unsqueeze(0), None

    data = np.fromfile(path, dtype=np.uint8)
    assert data.size == CAM_W * CAM_H * 3, \
        f"Expected {CAM_W * CAM_H * 3} bytes (640x360x3), got {data.size}"
    return preprocess(data.reshape(CAM_H, CAM_W, 3))


# ═══════════════════════════════════════════════════════════════════════════
# Postprocessing — Decode (미분 가능 영역)
# ═══════════════════════════════════════════════════════════════════════════

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def decode(d32, d16, conf_thresh=CONF_THRESHOLD):
    """Decode both heads: bbox + class scores.
    All operations here are differentiable (sigmoid, exp).

    Args:
        d32: float32 [30, 8, 14]
        d16: float32 [30, 16, 28]
        conf_thresh: objectness threshold

    Returns:
        list of {
            'bbox': (x1, y1, x2, y2),       # model input 좌표
            'obj_score': float,               # sigmoid(objectness)
            'class_scores': [5 floats],       # sigmoid(cls0~4)
            'class_id': int,                  # argmax of class_scores
            'class_name': str,
            'confidence': float,              # obj_score × max(class_scores)
        }
    """
    heads = [
        (d32, ANCHORS_S32),
        (d16, ANCHORS_S16),
    ]

    detections = []
    for head, anchors in heads:
        C, grid_h, grid_w = head.shape

        for a in range(NUM_ANCHORS):
            anchor_w, anchor_h = anchors[a]
            off = a * (NUM_CLASSES + 5)

            tx = head[off + 0]
            ty = head[off + 1]
            tw = head[off + 2]
            th = head[off + 3]
            obj_raw = head[off + 4]
            cls_raw = head[off + 5:off + 5 + NUM_CLASSES]

            obj_conf = sigmoid(obj_raw)

            for gy in range(grid_h):
                for gx in range(grid_w):
                    obj = float(obj_conf[gy, gx])
                    if obj < conf_thresh:
                        continue

                    # BBox decode (sigmoid + exp: 미분 가능)
                    cx = (gx + sigmoid(tx[gy, gx])) / grid_w
                    cy = (gy + sigmoid(ty[gy, gx])) / grid_h
                    bw = (anchor_w / MODEL_W) * np.exp(tw[gy, gx])
                    bh = (anchor_h / MODEL_H) * np.exp(th[gy, gx])

                    x1 = (cx - bw * 0.5) * MODEL_W
                    y1 = (cy - bh * 0.5) * MODEL_H
                    x2 = (cx + bw * 0.5) * MODEL_W
                    y2 = (cy + bh * 0.5) * MODEL_H

                    # Class decode (sigmoid: 미분 가능)
                    cls_scores = sigmoid(cls_raw[:, gy, gx]).tolist()
                    cls_id = int(np.argmax(cls_scores))
                    confidence = obj * cls_scores[cls_id]

                    detections.append({
                        'bbox': (float(x1), float(y1), float(x2), float(y2)),
                        'obj_score': obj,
                        'class_scores': cls_scores,
                        'class_id': cls_id,
                        'class_name': CLASS_NAMES[cls_id],
                        'confidence': confidence,
                    })

    return detections


# ═══════════════════════════════════════════════════════════════════════════
# Postprocessing — NMS (미분 불가, 선택적)
# ═══════════════════════════════════════════════════════════════════════════

def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = (a[2]-a[0]) * (a[3]-a[1])
    ab = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (aa + ab - inter) if (aa + ab - inter) > 0 else 0


def nms(detections, iou_thresh=IOU_THRESHOLD):
    """Per-class NMS. Non-differentiable.

    Args:
        detections: list of dicts from decode()
        iou_thresh: IoU threshold

    Returns:
        filtered list of dicts
    """
    results = []
    for cls_id in range(NUM_CLASSES):
        cls_dets = [d for d in detections if d['class_id'] == cls_id]
        cls_dets.sort(key=lambda d: d['confidence'], reverse=True)

        keep = []
        while cls_dets:
            best = cls_dets.pop(0)
            keep.append(best)
            cls_dets = [d for d in cls_dets if iou(best['bbox'], d['bbox']) < iou_thresh]

        results.extend(keep)

    return sorted(results, key=lambda d: d['confidence'], reverse=True)


# ═══════════════════════════════════════════════════════════════════════════
# Coordinate Mapping
# ═══════════════════════════════════════════════════════════════════════════

def map_to_raw(detections, resize_info):
    """Map bbox from model input space to raw frame coordinates."""
    if resize_info is None:
        return detections

    sx = resize_info['scale_x']
    sy = resize_info['scale_y']
    oh = resize_info['orig_h']
    ow = resize_info['orig_w']

    mapped = []
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        d = dict(d)
        d['bbox'] = (
            max(0, min(x1 / sx, ow)),
            max(0, min(y1 / sy, oh)),
            max(0, min(x2 / sx, ow)),
            max(0, min(y2 / sy, oh)),
        )
        mapped.append(d)

    return mapped


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Wyze EdgeAI 5-Class YOLO Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
모델 종류:
  ste      PyTorch STE. forward bit-exact + backward 가능. (기본값)
  numpy    원본 INT8 numpy 모델. bit-exact. backward 불가.
  diff     PyTorch float32 미분가능 근사. backward 가능.

예시:
  python classify.py frame.bin                              # raw 입력, STE 모델
  python classify.py frame.bin --model numpy                # numpy 모델
  python classify.py dump.bin --preprocessed --no-nms       # NMS 없이 decode까지만
""")
    parser.add_argument("input", help="Raw frame (640x360 RGB) or preprocessed (256x448x3 int8)")
    parser.add_argument("--model", default="ste",
                        choices=["numpy", "diff", "ste"],
                        help="Model variant (default: ste)")
    parser.add_argument("--preprocessed", action="store_true",
                        help="Input is preprocessed (256x448x3 int8, 344064 bytes)")
    parser.add_argument("--no-nms", action="store_true",
                        help="Skip NMS (decode only, 미분 가능 영역까지만)")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD,
                        help=f"Confidence threshold (default: {CONF_THRESHOLD})")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD,
                        help=f"IoU threshold for NMS (default: {IOU_THRESHOLD})")
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    model, model_type = load_model(args.model)
    print(f"  Type: {model_type}")

    # Load input
    input_chw, lb_info = load_input(args.input, args.preprocessed)
    input_type = "preprocessed" if args.preprocessed else "raw 640x360"
    print(f"Input: {args.input} ({input_type})")

    # Inference
    d32, d16 = run_inference(model, model_type, input_chw)
    print(f"Output: stride-32 {d32.shape}, stride-16 {d16.shape}")

    # Decode (미분 가능)
    detections = decode(d32, d16, args.conf)
    print(f"Decoded: {len(detections)} candidates (conf > {args.conf})")

    # NMS (선택)
    if not args.no_nms:
        detections = nms(detections, args.iou)
        print(f"After NMS: {len(detections)} detections")
    else:
        print(f"NMS skipped (--no-nms)")

    # Map to raw coordinates if applicable
    if lb_info:
        detections = map_to_raw(detections, lb_info)
        coord = "raw frame"
    else:
        coord = "model input"

    # Print results
    print(f"\nDetections ({coord}):")
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d['bbox']
        scores = " ".join(f"{CLASS_NAMES[j]}={d['class_scores'][j]:.3f}" for j in range(NUM_CLASSES))
        print(f"  [{i}] {d['class_name']} conf={d['confidence']:.3f} "
              f"obj={d['obj_score']:.3f} ({x1:.1f},{y1:.1f})-({x2:.1f},{y2:.1f})")
        print(f"       class_scores: {scores}")


if __name__ == '__main__':
    main()
