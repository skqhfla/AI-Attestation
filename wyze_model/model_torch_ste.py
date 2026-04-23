#!/usr/bin/env python3
"""
Wyze EdgeAI 5-Class YOLO — PyTorch STE (Straight-Through Estimator) version.
Forward: exact int operations (trunc, wrap, clamp) for high accuracy.
Backward: gradient passes through non-differentiable ops.

Usage:
    python model_torch_ste.py              # sanity check
    python model_torch_ste.py --compare    # compare with numpy model
    python model_torch_ste.py --test-grad  # verify gradient flows
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_BLOB_PATH = SCRIPT_DIR / "model_blob"
WEIGHT_BLOB_PATH = SCRIPT_DIR / "weight_blob"

LOAD_PARAM_SIZE = {
    16: 12, 33: 4, 46: 76, 51: 68, 53: 72,
    67: 4, 69: 20, 71: 4, 73: 72, 74: 72, 75: 48, 76: 0,
}


# ═══════════════════════════════════════════════════════════════════════════
# STE: forward = exact, backward = identity
# ═══════════════════════════════════════════════════════════════════════════

class _STETrunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.trunc(x)
    @staticmethod
    def backward(ctx, grad):
        return grad

class _STEWrap16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return ((x + 32768) % 65536) - 32768
    @staticmethod
    def backward(ctx, grad):
        return grad

def ste_trunc(x):
    return _STETrunc.apply(x)

def ste_wrap16(x):
    return _STEWrap16.apply(x)


# ═══════════════════════════════════════════════════════════════════════════
# Weight decoders (compat=5)
# ═══════════════════════════════════════════════════════════════════════════

_SPATIAL_PAIRS = [
    ((0,0),(0,1)), ((0,2),(1,0)), ((1,1),(1,2)),
    ((2,0),(2,1)), (None,(2,2)),
]

def _decode_dw(raw, ch):
    k = np.zeros((ch, 3, 3), dtype=np.int8)
    for g in range(ch//16):
        gr = raw[g*160:(g+1)*160]; cb = g*16
        for pi, (pA, pB) in enumerate(_SPATIAL_PAIRS):
            vl = gr[pi*32:pi*32+16]; vh = gr[pi*32+16:pi*32+32]
            for j in range(8):
                if pA: k[cb+j,pA[0],pA[1]]=vl[2*j]; k[cb+8+j,pA[0],pA[1]]=vh[2*j]
                if pB: k[cb+j,pB[0],pB[1]]=vl[2*j+1]; k[cb+8+j,pB[0],pB[1]]=vh[2*j+1]
    return k

def _decode_c1x1(raw, oc, ic):
    w = np.zeros((oc, ic), dtype=np.int8)
    for og in range(oc//16):
        for ih in range(ic//2):
            for ob in range(2):
                off = og*ic*16+ih*32+ob*16
                for j in range(8):
                    w[og*16+ob*8+j,ih*2]=raw[off+2*j]; w[og*16+ob*8+j,ih*2+1]=raw[off+2*j+1]
    return w

def _decode_convv2(raw, oc, ic):
    w = np.zeros((oc, ic, 3, 3), dtype=np.int8)
    for ky in range(3):
        for kx in range(3):
            for c in range(ic):
                br = kx*ic+c; hw = br//2; bp = br%2
                for o in range(oc):
                    w[o,c,ky,kx] = raw[ky*160+hw*32+(o//8)*16+(o%8)*2+bp]
    return w


# ═══════════════════════════════════════════════════════════════════════════
# STE Requant layers
# ═══════════════════════════════════════════════════════════════════════════

class STEConvMULW(nn.Module):
    """nn.Conv2d + STE MULW requant (Conv1x1V3)."""
    def __init__(self, conv, scl_in, sraw_adj, srarh_sh, zp, upper):
        super().__init__()
        self.conv = conv
        self.register_buffer('scl_in', scl_in)
        self.register_buffer('sraw_shift', ((15 - sraw_adj.int()) & 0x1F).float())
        self.register_buffer('srarh_shift', (srarh_sh.int() & 0xF).float())
        self.zp = zp; self.upper = upper

    def forward(self, x):
        acc = self.conv(x)
        acc = ste_wrap16(acc)
        product = acc * self.scl_in.view(1, -1, 1, 1)
        shifted = ste_trunc(product / (2.0 ** self.sraw_shift.view(1, -1, 1, 1)))
        sat = torch.clamp(shifted, -32768, 32767)
        sh = self.srarh_shift.view(1, -1, 1, 1)
        half = torch.where(sh > 0, 2.0 ** (sh - 1), torch.zeros_like(sh))
        rounded = ste_trunc((sat + half) / (2.0 ** sh))
        return torch.clamp(rounded, 0, self.upper) - self.zp


class STEConvMULQH(nn.Module):
    """nn.Conv2d + STE MULQH requant (DWv3 compat=5)."""
    def __init__(self, conv, scl_in, srarh_sh, zp, upper, pad_fill):
        super().__init__()
        self.conv = conv
        self.register_buffer('scl_in', scl_in)
        self.register_buffer('srarh_shift', (srarh_sh.int() & 0xF).float())
        self.zp = zp; self.upper = upper; self.pad_fill = pad_fill

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), value=self.pad_fill)
        acc = F.conv2d(x, self.conv.weight, self.conv.bias, padding=0, groups=self.conv.groups)
        acc = ste_wrap16(acc)
        prod = acc * self.scl_in.view(1, -1, 1, 1)
        shifted = torch.clamp(prod * 2, -2147483648, 2147483647)
        mq = ste_trunc(shifted / 65536)
        sh = self.srarh_shift.view(1, -1, 1, 1)
        half = torch.where(sh > 0, 2.0 ** (sh - 1), torch.zeros_like(sh))
        rounded = ste_trunc((mq + half) / (2.0 ** sh))
        return torch.clamp(rounded, 0, self.upper) - self.zp


class STEConvMULQW(nn.Module):
    """nn.Conv2d + STE MULQW requant (ConvV2)."""
    def __init__(self, conv, bias_f, scale0, scale2, zp, upper, pad_fill, stride):
        super().__init__()
        self.conv = conv
        self.register_buffer('bias_f', bias_f)
        self.register_buffer('scale0', scale0)
        self.register_buffer('scale2', scale2)
        self.zp = zp; self.upper = upper; self.pad_fill = pad_fill; self.stride = stride

    def forward(self, x):
        s = self.stride; H, W = x.shape[2], x.shape[3]
        Ho, Wo = -(-H//s), -(-W//s)
        ph = max((Ho-1)*s+3-H, 0); pw = max((Wo-1)*s+3-W, 0)
        x = F.pad(x, (0, pw, 0, ph), value=self.pad_fill)
        acc = F.conv2d(x, self.conv.weight, None, stride=s, padding=0)
        acc = ste_wrap16(acc)
        biased = acc + self.bias_f.view(1, -1, 1, 1)
        product = biased * self.scale0.view(1, -1, 1, 1)
        shifted = torch.clamp(product * 2, -(2**63), 2**63 - 1)
        mulqw = ste_trunc(shifted / (2**32))
        sh = self.scale2.view(1, -1, 1, 1).float()
        half = torch.where(sh > 0, 2.0 ** (sh - 1), torch.zeros_like(sh))
        rounded = ste_trunc((mulqw + half) / (2.0 ** sh))
        return torch.clamp(rounded, 0, self.upper) - self.zp


class STEConvMP2(nn.Module):
    def __init__(self, qconv):
        super().__init__()
        self.qconv = qconv
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        return self.pool(self.qconv(x))


class DetectHead(nn.Module):
    def __init__(self, conv, float_scale):
        super().__init__()
        self.conv = conv
        self.register_buffer('float_scale', float_scale)
    def forward(self, x):
        return self.conv(x) * self.float_scale.view(1, -1, 1, 1)


# ═══════════════════════════════════════════════════════════════════════════
# Network
# ═══════════════════════════════════════════════════════════════════════════

class WyzeClassifySTE(nn.Module):
    INPUT_H = 256
    INPUT_W = 448

    def __init__(self):
        super().__init__()
        self._parse_and_build()

    def _parse_and_build(self):
        mb = MODEL_BLOB_PATH.read_bytes()
        wb = WEIGHT_BLOB_PATH.read_bytes()

        w0 = struct.unpack_from('<I', mb, 24)[0]
        w1 = struct.unpack_from('<I', mb, 28)[0]
        num_layers = w0 & 0xFFFFFF
        self.num_blobs = w1 & 0xFFFFFF

        off = 32
        self.layer_defs = []
        for i in range(num_layers):
            t = struct.unpack_from('<I', mb, off)[0]
            ni = struct.unpack_from('<I', mb, off+4)[0]
            no = struct.unpack_from('<I', mb, off+8)[0]
            off += 12
            ins = [struct.unpack_from('<I', mb, off+j*4)[0] for j in range(ni)]
            off += ni*4
            outs = [struct.unpack_from('<I', mb, off+j*4)[0] for j in range(no)]
            off += no*4
            lps = LOAD_PARAM_SIZE[t]
            params = [struct.unpack_from('<i', mb, off+j*4)[0] for j in range(lps//4)]
            off += lps
            self.layer_defs.append({'idx':i,'type':t,'inputs':ins,'outputs':outs,'params':params})

        ptr = 0
        def read_i8(n):
            nonlocal ptr; d=np.frombuffer(wb[ptr:ptr+n],dtype=np.int8).copy(); ptr+=n; return d
        def read_i16(n):
            nonlocal ptr; d=np.frombuffer(wb[ptr:ptr+2*n],dtype=np.int16).copy(); ptr+=2*n; return d
        def read_i32(n):
            nonlocal ptr; d=np.frombuffer(wb[ptr:ptr+4*n],dtype=np.int32).copy(); ptr+=4*n; return d
        def read_f32(n):
            nonlocal ptr; d=np.frombuffer(wb[ptr:ptr+4*n],dtype=np.float32).copy(); ptr+=4*n; return d

        self.layer_modules = nn.ModuleList()
        self._module_map = {}

        for ldef in self.layer_defs:
            t = ldef['type']; p = ldef['params']; module = None

            if t == 46:
                oc, wbs = p[0], p[12]
                w = _decode_convv2(read_i8(wbs), oc, 3).astype(np.float32)
                bias = read_i32(oc); s0 = read_i32(oc); s2 = read_i32(oc)
                pf = float(-(1<<(p[15]-1)))
                conv = nn.Conv2d(3, oc, 3, stride=p[5], padding=0, bias=False)
                conv.weight.data = torch.from_numpy(w)
                module = STEConvMULQW(conv,
                    torch.from_numpy(bias.astype(np.float32)),
                    torch.from_numpy(s0.astype(np.float32)),
                    torch.from_numpy(s2.astype(np.float32)),
                    float((1<<p[16])//2), float((1<<p[16])-1), pf, p[5])

            elif t == 73:
                oc, wbs = p[0], p[12]
                k = _decode_dw(read_i8(wbs), oc).astype(np.float32).reshape(oc,1,3,3)
                bias = read_i16(oc); scl = read_i16(oc); sr = read_i16(oc); sa = read_i16(oc)
                pf = float(-(1<<(p[14]-1)))
                conv = nn.Conv2d(oc, oc, 3, padding=0, groups=oc, bias=True)
                conv.weight.data = torch.from_numpy(k)
                conv.bias.data = torch.from_numpy(bias.astype(np.float32))
                module = STEConvMULQH(conv,
                    torch.from_numpy(scl.astype(np.float32)),
                    torch.from_numpy(sr.astype(np.float32)),
                    float((1<<p[15])//2), float((1<<p[15])-1), pf)

            elif t in (53, 74):
                oc, wbs = p[0], p[12]; ic = wbs//oc
                w = _decode_c1x1(read_i8(wbs), oc, ic).astype(np.float32).reshape(oc,ic,1,1)
                bias = read_i16(oc); scl = read_i16(oc); sr = read_i16(oc); sa = read_i16(oc)
                conv = nn.Conv2d(ic, oc, 1, bias=True)
                conv.weight.data = torch.from_numpy(w)
                conv.bias.data = torch.from_numpy(bias.astype(np.float32))
                qconv = STEConvMULW(conv,
                    torch.from_numpy(scl.astype(np.float32)),
                    torch.from_numpy(sa.astype(np.float32)),
                    torch.from_numpy(sr.astype(np.float32)),
                    float((1<<p[15])//2), float((1<<p[15])-1))
                module = STEConvMP2(qconv) if t == 74 else qconv

            elif t == 51:
                oc, wbs = p[0], p[12]; ic = wbs//oc
                w = read_i8(wbs).reshape(oc,ic,1,1).astype(np.float32)
                bias = read_i32(oc); scale = read_f32(oc)
                conv = nn.Conv2d(ic, oc, 1, bias=True)
                conv.weight.data = torch.from_numpy(w)
                conv.bias.data = torch.from_numpy(bias.astype(np.float32))
                module = DetectHead(conv, torch.from_numpy(scale))

            if module is not None:
                self._module_map[ldef['idx']] = len(self.layer_modules)
                self.layer_modules.append(module)

        assert ptr == len(wb)

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(0)
        blobs = [None] * self.num_blobs

        for ldef in self.layer_defs:
            t = ldef['type']; ins = ldef['inputs']; outs = ldef['outputs']
            if t == 16:
                result = x
            elif t == 33:
                result = blobs[ins[0]]
                for o in outs: blobs[o] = result
                continue
            elif t == 71:
                result = torch.cat([blobs[i] for i in ins], dim=1)
            elif t == 75:
                result = F.max_pool2d(blobs[ins[0]], 2)
            elif t == 69:
                inp = blobs[ins[0]]; p = ldef['params']
                fill = float(-(1 << (p[1] - 1)))
                B, C, H, W = inp.shape
                out = torch.full((B, C, H*p[2], W*p[3]), fill, dtype=inp.dtype, device=inp.device)
                out[:, :, ::p[2], ::p[3]] = inp
                result = out
            elif t == 76:
                return blobs[ins[0]].squeeze(0), blobs[ins[1]].squeeze(0)
            else:
                mid = self._module_map.get(ldef['idx'])
                if mid is not None:
                    result = self.layer_modules[mid](blobs[ins[0]])
                else:
                    raise ValueError(f"Unhandled type {t}")
            for o in outs: blobs[o] = result

        raise RuntimeError("Gather not found")


def main():
    if '--test-grad' in sys.argv:
        print("Loading STE model...")
        model = WyzeClassifySTE(); model.eval()
        x = torch.randn(1, 3, 256, 448, requires_grad=True)
        d32, d16 = model(x)
        loss = d32[4, 4, 7]
        loss.backward()
        grad = x.grad
        print(f"Gradient: min={grad.min():.6f} max={grad.max():.6f} "
              f"nonzero={(grad!=0).sum().item()}/{grad.numel()}")
        if grad.abs().sum() > 0:
            print("Gradient flows. Adversarial attacks possible.")
        else:
            print("WARNING: zero gradients!")
        return

    if '--compare' in sys.argv:
        from model import WyzeClassify
        print("Loading STE model...")
        pt_model = WyzeClassifySTE(); pt_model.eval()
        print("Loading numpy model...")
        np_model = WyzeClassify.load()

        dump_dir = SCRIPT_DIR.parent / "dumps_model"
        files = sorted(dump_dir.glob("frame_*_input.bin"))
        print(f"Comparing {len(files)} frames...\n")

        exact = 0; worst_mad = 0
        for i, f in enumerate(files):
            x_np = np.fromfile(f, dtype=np.int8).reshape(256, 448, 3).transpose(2, 0, 1)
            d32_np, d16_np = np_model.forward(x_np)
            x_t = torch.from_numpy(x_np.astype(np.float32))
            with torch.no_grad():
                d32_pt, d16_pt = pt_model(x_t)
            mad = max(np.abs(d32_np - d32_pt.numpy()).mean(),
                      np.abs(d16_np - d16_pt.numpy()).mean())
            if mad == 0: exact += 1
            worst_mad = max(worst_mad, mad)
            print(f"  [{i+1}/{len(files)}] MAD={mad:.6f}")

        print(f"\nPerfect match: {exact}/{len(files)} ({100*exact/len(files):.1f}%)")
        print(f"Worst MAD: {worst_mad:.6f}")
        return

    print("Loading STE model...")
    model = WyzeClassifySTE(); model.eval()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.zeros(1, 3, 256, 448)
    with torch.no_grad():
        d32, d16 = model(x)
    print(f"head_s32: {d32.shape}  head_s16: {d16.shape}")
    print("OK. Run --compare or --test-grad")


if __name__ == '__main__':
    main()
