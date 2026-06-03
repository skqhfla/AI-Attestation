#!/usr/bin/env python3
"""
Benchmark variant of generate_challenge.py.

원본 동작을 그대로 유지하되 자원 측정용으로 다음을 노출:
  argv[1]: model_name      (default: resnet20_quan)
  argv[2]: worker_id       (default: 0)  -- 병렬 실행 시 save_dir 충돌 방지
  argv[3]: num_challenges  (default: 100)

CPU/GPU 전환은 환경변수 CUDA_VISIBLE_DEVICES 로 제어:
  CUDA_VISIBLE_DEVICES=''   python generate_challenge_bench.py ...   # CPU
  CUDA_VISIBLE_DEVICES=0    python generate_challenge_bench.py ...   # GPU0
"""

import json
import os
import re
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

import models  # noqa: F401  (resolve package path)
from models.quan_resnet_cifar import resnet20_quan
from models.quan_vgg_cifar import vgg11_bn_quan
from models.quan_wideresnet import wideresnet_quan
from models.quantization import quan_Conv2d, quan_Linear


def quantize(model):
    """attestation/generate_challenge.py 와 동일.
    Conv2d / Linear 를 quan_* 버전으로 in-place 치환."""
    for name, child in model.named_children():
        if isinstance(child, nn.Conv2d):
            ql = quan_Conv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
            )
            ql.weight.data = child.weight.data.clone()
            if child.bias is not None:
                ql.bias.data = child.bias.data.clone()
            setattr(model, name, ql)
        elif isinstance(child, nn.Linear):
            ql = quan_Linear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
            )
            ql.weight.data = child.weight.data.clone()
            if child.bias is not None:
                ql.bias.data = child.bias.data.clone()
            setattr(model, name, ql)
        else:
            quantize(child)


# ── 인자 파싱 ───────────────────────────────────────────────────────────────
model_name     = sys.argv[1] if len(sys.argv) > 1 else 'resnet20_quan'
worker_id      = int(sys.argv[2]) if len(sys.argv) > 2 else 0
num_challenges = int(sys.argv[3]) if len(sys.argv) > 3 else 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[w{worker_id}] model={model_name}  device={device}  N={num_challenges}",
      flush=True)

# ── 모델 로드 ───────────────────────────────────────────────────────────────
num_classes = 10
image_size = (32, 32)
save_root = './save/2025-09-09'

if model_name == 'resnet20_quan':
    model = resnet20_quan(num_classes=num_classes)
    model_path = 'resnet20_cifar10_best_weights'
elif model_name == 'vgg11_quan':
    model = vgg11_bn_quan(num_classes=num_classes)
    model_path = 'vgg11_cifar10_best_weights'
elif model_name == 'wideresnet_quan':
    model = wideresnet_quan(depth=28, num_classes=num_classes)
    model_path = 'wideresnet_cifar10_best_weights'
elif model_name == "efficientnetv2_quan":
    # attestation/generate_challenge.py 와 동일 로딩.
    # torchvision efficientnet_v2_l → classifier 교체 → quantize() in-place.
    # CIFAR-100, 224x224 입력.
    from torchvision.models import efficientnet_v2_l
    num_classes = 100
    image_size = (224, 224)
    model = efficientnet_v2_l()
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.25, inplace=True),
        nn.Linear(model.classifier[-1].in_features, num_classes),
    )
    quantize(model)
    model_path = 'efficientnetv2_cifar100_best_weights'
elif model_name == "tresnet_quan":
    # attestation/generate_challenge.py 와 동일 로딩.
    # timm tresnet_l (pretrained) → quantize() in-place.
    # CIFAR-100, 224x224 입력.
    import timm
    num_classes = 100
    image_size = (224, 224)
    model = timm.create_model('tresnet_l', pretrained=True, num_classes=num_classes)
    quantize(model)
    model_path = 'tresnet_cifar100_best_weights'
else:
    raise ValueError(f"지원되지 않는 모델 이름: {model_name}")

model = model.to(device).eval()
checkpoint_path = f"{save_root}/{model_path}.pth"
ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
print(f"[w{worker_id}] checkpoint loaded: {checkpoint_path}", flush=True)


def generate_random_image():
    return torch.rand((1, 3, image_size[0], image_size[1]), device=device)


def save_image(tensor, filename):
    tensor = tensor.squeeze(0).detach().cpu()
    image = transforms.ToPILImage()(tensor)
    image.save(filename)


# ── 저장 디렉터리 (worker_id 로 분리해 병렬 충돌 방지) ──────────────────────
save_dir = f"./data/challenge_bench/{model_name}_w{worker_id}/random"
os.makedirs(save_dir, exist_ok=True)

# 기존 파일 카운트 (sweep 반복 시 누적 방지를 위해 사전 청소 권장)
existing = [f for f in os.listdir(save_dir)
            if re.match(r"challenge_image_(\d+)\.png", f)]
max_idx = max([int(re.match(r"challenge_image_(\d+)\.png", f).group(1))
               for f in existing], default=0)
remaining = max(0, num_challenges - len(existing))
print(f"[w{worker_id}] generating {remaining} images "
      f"(existing={len(existing)}, target={num_challenges})",
      flush=True)

# ── 챌린지 생성 루프 ───────────────────────────────────────────────────────
t0 = time.time()
for i in range(remaining):
    img = generate_random_image()
    save_image(img, os.path.join(save_dir, f'challenge_image_{max_idx + i + 1}.png'))
t_gen = time.time() - t0
print(f"[w{worker_id}] generation done in {t_gen:.2f}s", flush=True)

# ── 마지막 batch forward (피크 메모리 발생 구간) ───────────────────────────
def load_and_predict():
    tfm = transforms.Compose([transforms.ToTensor()])
    paths = sorted(os.path.join(save_dir, f) for f in os.listdir(save_dir)
                   if f.endswith('.png'))
    imgs = [tfm(Image.open(p)).unsqueeze(0) for p in paths]
    batch = torch.cat(imgs, dim=0).to(device)
    with torch.no_grad():
        out = model(batch)
        probs = F.softmax(out, dim=1)
    topk_vals, topk_idx = torch.topk(probs, k=min(10, num_classes), dim=1)
    return batch.shape, topk_vals.shape

t1 = time.time()
shape, _ = load_and_predict()
t_pred = time.time() - t1
t_total = time.time() - t0
print(f"[w{worker_id}] predict done in {t_pred:.2f}s  batch={shape}", flush=True)
print(f"[w{worker_id}] TOTAL {t_total:.2f}s", flush=True)

# ── sweep_resources.py 가 읽어가는 timing 파일 ─────────────────────────────
# save_dir 의 부모(./data/challenge_bench/<model>_w<id>/) 에 저장.
timing = {
    'worker_id': worker_id,
    'model_name': model_name,
    'num_challenges': num_challenges,
    'gen_time_s': round(t_gen, 3),
    'pred_time_s': round(t_pred, 3),
    'total_time_s': round(t_total, 3),
    'batch_shape': list(shape),
}
timing_path = os.path.join(os.path.dirname(save_dir), '_timing.json')
try:
    with open(timing_path, 'w') as f:
        json.dump(timing, f)
    print(f"[w{worker_id}] timing → {timing_path}", flush=True)
except OSError as e:
    print(f"[w{worker_id}] WARN: timing write failed: {e}", flush=True)
