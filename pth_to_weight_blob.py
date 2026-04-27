#!/usr/bin/env python3
"""
pth_to_weight_blob.py
======================
BFA 로 변조된 PyTorch state_dict (.pth, WyzeClassifySTE 가 저장한 것) 를
디바이스가 직접 읽는 원본 ``weight_blob`` 바이너리 포맷으로 되돌린다.

목적
  - PyTorch 측 BFA 결과를 실제 Wyze 디바이스에 deploy 가능한 형태로 변환.
  - encoder 4 종을 ``model_torch_ste`` 의 decoder 와 정확히 역으로 구현하므로,
    정상 동작 시 ``원본 wb -> decoder -> .pth -> encoder -> 새 wb -> decoder``
    경로의 round-trip 이 bit-exact 함.

가정
  - BFA 는 ``Conv2d.weight`` 만 비트를 뒤집는다 (``attack_wyze/BFA.py`` 의 mask
    가 cls_channels 의 weight 에만 걸려있음). bias, scale 등 다른 buffer 는 원본
    ``weight_blob`` 에서 그대로 byte 복사.
  - LOAD_PARAM_SIZE 안에 정의된 layer type 가운데 weight 를 갖는 것은
    {46, 73, 53, 74, 51} 4 종. 그 외 layer 는 weight_blob 을 read 하지 않으므로
    insert 도 하지 않는다 (decoder 와 동일 행동).

사용
  단일 변환:
      python pth_to_weight_blob.py results/wyze/weights/wyze_bfa_bit5_conf0.4xx.pth \
          --output results/wyze/weight_blob/wyze_bfa_bit5.weight_blob

  변환 + round-trip 검증:
      python pth_to_weight_blob.py <pth> --output <wb> --verify

  검증용 샘플 입력 지정 (기본: neutral gray):
      python pth_to_weight_blob.py <pth> --output <wb> --verify \
          --sample_input dumps_raw/frame_0000_input.bin
"""

import argparse
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
WYZE_DIR   = SCRIPT_DIR / "wyze_model"
sys.path.insert(0, str(WYZE_DIR))

from model_torch_ste import (    # noqa: E402
    WyzeClassifySTE,
    MODEL_BLOB_PATH,
    WEIGHT_BLOB_PATH,
    LOAD_PARAM_SIZE,
    _SPATIAL_PAIRS,
)


# ────────────────────────────────────────────────────────────────────────────
# Encoders — model_torch_ste 의 _decode_* 와 정확히 역연산
# ────────────────────────────────────────────────────────────────────────────

def _encode_dw(k, ch):
    """type 73 (depthwise 3x3) 가중치 (ch, 3, 3) int8 → packed bytes (ch*10).
    decoder ``_decode_dw`` 의 반복문을 그대로 뒤집은 것.
    """
    raw = np.zeros(ch * 10, dtype=np.int8)
    for g in range(ch // 16):
        gr = g * 160
        cb = g * 16
        for pi, (pA, pB) in enumerate(_SPATIAL_PAIRS):
            vl = gr + pi * 32
            vh = vl + 16
            for j in range(8):
                if pA:
                    raw[vl + 2*j]     = k[cb + j,     pA[0], pA[1]]
                    raw[vh + 2*j]     = k[cb + 8 + j, pA[0], pA[1]]
                if pB:
                    raw[vl + 2*j + 1] = k[cb + j,     pB[0], pB[1]]
                    raw[vh + 2*j + 1] = k[cb + 8 + j, pB[0], pB[1]]
    return raw.tobytes()


def _encode_c1x1(w, oc, ic):
    """type 53/74 (1x1 conv) 가중치 (oc, ic) int8 → packed bytes (oc*ic).
    decoder ``_decode_c1x1`` 의 인덱싱을 그대로 뒤집는다.
    """
    raw = np.zeros(oc * ic, dtype=np.int8)
    for og in range(oc // 16):
        for ih in range(ic // 2):
            for ob in range(2):
                off = og * ic * 16 + ih * 32 + ob * 16
                for j in range(8):
                    raw[off + 2*j]     = w[og * 16 + ob * 8 + j, ih * 2]
                    raw[off + 2*j + 1] = w[og * 16 + ob * 8 + j, ih * 2 + 1]
    return raw.tobytes()


def _encode_convv2(w, oc, ic, wbs):
    """type 46 (first conv 3x3, 3-ch input) 가중치 (oc, ic, 3, 3) int8.

    decoder 가 ``raw[ky*160 + hw*32 + (o//8)*16 + (o%8)*2 + bp]`` 로 읽는 패턴을
    그대로 역으로 채운다. ``wbs`` 는 layer params[12] 에서 받은 원본 버퍼 크기.
    """
    raw = np.zeros(wbs, dtype=np.int8)
    for ky in range(3):
        for kx in range(3):
            for c in range(ic):
                br = kx * ic + c
                hw = br // 2
                bp = br % 2
                for o in range(oc):
                    raw[ky * 160 + hw * 32 + (o // 8) * 16 + (o % 8) * 2 + bp] = (
                        w[o, c, ky, kx]
                    )
    return raw.tobytes()


# ────────────────────────────────────────────────────────────────────────────
# model_blob 파싱 — WyzeClassifySTE._parse_and_build 의 layer_defs 부분만 추출
# ────────────────────────────────────────────────────────────────────────────

def parse_layer_defs(model_blob_path: Path):
    mb = model_blob_path.read_bytes()
    w0 = struct.unpack_from('<I', mb, 24)[0]
    num_layers = w0 & 0xFFFFFF

    off = 32
    defs = []
    for i in range(num_layers):
        t = struct.unpack_from('<I', mb, off)[0]
        ni = struct.unpack_from('<I', mb, off + 4)[0]
        no = struct.unpack_from('<I', mb, off + 8)[0]
        off += 12 + 4 * ni + 4 * no
        lps = LOAD_PARAM_SIZE[t]
        params = [
            struct.unpack_from('<i', mb, off + j * 4)[0]
            for j in range(lps // 4)
        ]
        off += lps
        defs.append({'idx': i, 'type': t, 'params': params})
    return defs


# ────────────────────────────────────────────────────────────────────────────
# Conversion
# ────────────────────────────────────────────────────────────────────────────

def _to_int8(t):
    """conv weight 가 float32 로 저장돼 있어도 실제 값은 정수. 안전하게 INT8 변환."""
    arr = t.detach().cpu().numpy()
    out = np.clip(np.round(arr), -128, 127).astype(np.int8)
    return out


def convert(pth_path: Path,
            output_wb_path: Path,
            model_blob_path: Path = MODEL_BLOB_PATH,
            original_wb_path: Path = WEIGHT_BLOB_PATH,
            verbose: bool = True):
    """BFA-modified .pth → weight_blob bytes.

    weight 부분만 .pth 의 값으로 re-encode 하고, bias / scale / 그 외 buffer 는
    원본 weight_blob 의 byte 를 그대로 복사한다.
    """
    if verbose:
        print(f"Loading state_dict from {pth_path.name} ...")
    sd = torch.load(pth_path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']

    # 원본 weight_blob 으로 모델 빌드 후 BFA state_dict 덮어쓰기.
    # strict=False: BFA 가 만든 .pth 에 추가/누락 키가 있을 수 있어 안전하게.
    model = WyzeClassifySTE()
    res = model.load_state_dict(sd, strict=False)
    if verbose and (res.missing_keys or res.unexpected_keys):
        print(f"  load_state_dict: missing={len(res.missing_keys)} "
              f"unexpected={len(res.unexpected_keys)}")
    model.eval()

    layer_defs = parse_layer_defs(model_blob_path)
    wb_orig = original_wb_path.read_bytes()

    new_wb = bytearray()
    ptr = 0
    n_layers_encoded = 0

    for ldef in layer_defs:
        t = ldef['type']
        p = ldef['params']
        # 모델 빌드 시 등록된 layer 인지 확인.
        mod_idx = model._module_map.get(ldef['idx'])

        if t == 46:
            oc = p[0]
            ic = 3
            wbs = p[12]
            module = model.layer_modules[mod_idx]
            w = _to_int8(module.conv.weight)             # (oc, ic, 3, 3)
            new_wb.extend(_encode_convv2(w, oc, ic, wbs))
            ptr += wbs
            extra = oc * 4 + oc * 4 + oc * 4  # bias(i32), s0(i32), s2(i32)
            new_wb.extend(wb_orig[ptr:ptr + extra]); ptr += extra
            n_layers_encoded += 1

        elif t == 73:
            oc = p[0]
            wbs = p[12]
            module = model.layer_modules[mod_idx]
            w = _to_int8(module.conv.weight).reshape(oc, 3, 3)
            new_wb.extend(_encode_dw(w, oc))
            ptr += wbs
            extra = oc * 2 * 4  # bias, scl, sr, sa  (i16 each, oc 길이)
            new_wb.extend(wb_orig[ptr:ptr + extra]); ptr += extra
            n_layers_encoded += 1

        elif t in (53, 74):
            oc = p[0]
            wbs = p[12]
            ic = wbs // oc
            module = model.layer_modules[mod_idx]
            qconv = module.qconv if t == 74 else module
            w = _to_int8(qconv.conv.weight).reshape(oc, ic)
            new_wb.extend(_encode_c1x1(w, oc, ic))
            ptr += wbs
            extra = oc * 2 * 4
            new_wb.extend(wb_orig[ptr:ptr + extra]); ptr += extra
            n_layers_encoded += 1

        elif t == 51:
            oc = p[0]
            wbs = p[12]
            ic = wbs // oc
            module = model.layer_modules[mod_idx]
            w = _to_int8(module.conv.weight).reshape(oc, ic)
            new_wb.extend(w.tobytes())
            ptr += wbs
            extra = oc * 4 + oc * 4  # bias(i32), scale(f32)
            new_wb.extend(wb_orig[ptr:ptr + extra]); ptr += extra
            n_layers_encoded += 1

        # 그 외 layer 는 weight_blob 을 read 하지 않으므로 건너뜀.

    if ptr != len(wb_orig):
        raise RuntimeError(
            f"weight_blob layout mismatch: rebuilt up to byte {ptr}, "
            f"but original weight_blob is {len(wb_orig)} bytes. "
            f"Encoder is out of sync with decoder; check LOAD_PARAM_SIZE / "
            f"per-type buffer reads."
        )
    if len(new_wb) != len(wb_orig):
        raise RuntimeError(
            f"output length mismatch: {len(new_wb)} vs original {len(wb_orig)}"
        )

    output_wb_path.parent.mkdir(parents=True, exist_ok=True)
    output_wb_path.write_bytes(bytes(new_wb))
    if verbose:
        print(f"Wrote {output_wb_path}  "
              f"({len(new_wb)} bytes, {n_layers_encoded} weight layers)")
    return output_wb_path


# ────────────────────────────────────────────────────────────────────────────
# Round-trip verification
# ────────────────────────────────────────────────────────────────────────────

def verify(pth_path: Path, converted_wb_path: Path,
           sample_input_path: Path = None):
    """변환된 weight_blob 으로 다시 모델을 빌드해 forward 결과가
    .pth 를 직접 적용한 모델의 결과와 일치하는지 확인.

    bit-exact 라면 d32, d16 의 max |diff| 가 0 (또는 부동소수점 오차 수준).
    """
    print("\n=== Round-trip verification ===")

    sd = torch.load(pth_path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']

    ref = WyzeClassifySTE()
    ref.load_state_dict(sd, strict=False)
    ref.eval()

    if sample_input_path and sample_input_path.exists():
        from classify import load_input
        x_int8, _ = load_input(str(sample_input_path), preprocessed=False)
        x = x_int8.squeeze(0)
        print(f"  sample input: {sample_input_path.name}")
    else:
        x = torch.zeros(3, WyzeClassifySTE.INPUT_H, WyzeClassifySTE.INPUT_W)
        print("  sample input: zeros (neutral)")

    with torch.no_grad():
        d32_ref, d16_ref = ref(x)

    # 새 weight_blob 으로 모델 빌드 — WEIGHT_BLOB_PATH 를 임시로 swap.
    backup = WEIGHT_BLOB_PATH.with_name(WEIGHT_BLOB_PATH.name + ".bak")
    shutil.copy(WEIGHT_BLOB_PATH, backup)
    shutil.copy(converted_wb_path, WEIGHT_BLOB_PATH)
    try:
        rebuilt = WyzeClassifySTE()
        rebuilt.eval()
        with torch.no_grad():
            d32_rt, d16_rt = rebuilt(x)
    finally:
        shutil.copy(backup, WEIGHT_BLOB_PATH)
        backup.unlink()

    diff32 = (d32_ref - d32_rt).abs().max().item()
    diff16 = (d16_ref - d16_rt).abs().max().item()
    print(f"  d32 max|ref - rt| = {diff32:.6f}")
    print(f"  d16 max|ref - rt| = {diff16:.6f}")
    if max(diff32, diff16) < 1e-4:
        print("  [OK]   round-trip is bit-exact within fp tolerance")
    else:
        print("  [FAIL] mismatch — encoder may diverge from decoder")
    return diff32, diff16


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('pth',
                        help='BFA-modified .pth (state_dict 또는 checkpoint dict)')
    parser.add_argument('--output', '-o', required=True,
                        help='출력 weight_blob 파일 경로')
    parser.add_argument('--model_blob', default=str(MODEL_BLOB_PATH),
                        help=f'(default: {MODEL_BLOB_PATH})')
    parser.add_argument('--original_wb', default=str(WEIGHT_BLOB_PATH),
                        help=f'bias/scale 을 복사해 올 원본 weight_blob '
                             f'(default: {WEIGHT_BLOB_PATH})')
    parser.add_argument('--verify', action='store_true',
                        help='변환 후 round-trip forward 검증')
    parser.add_argument('--sample_input', default=None,
                        help='--verify 시 사용할 샘플 .bin 입력 (선택)')
    args = parser.parse_args()

    output = convert(
        Path(args.pth),
        Path(args.output),
        Path(args.model_blob),
        Path(args.original_wb),
    )
    if args.verify:
        verify(Path(args.pth), output,
               Path(args.sample_input) if args.sample_input else None)


if __name__ == '__main__':
    main()
