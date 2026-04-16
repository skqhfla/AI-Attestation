import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re
from PIL import Image
import numpy as np
import time
import torchvision.transforms as transforms

# 모델을 불러오기 위한 models 모듈 사용 (main.py와 동일 경로에 있다고 가정)
import models  # main.py와 동일한 경로에 있는 models 모듈 불러오기

# ResNet20 모델을 CIFAR-10에 맞춰 불러오기 (models 모듈 사용) !!!!!!!!!!!!!!!!!!!! 수정 
from models.quan_resnet_cifar import resnet20_quan
from models.quan_resnet_imagenet import resnet18_quan
from models.quan_vgg_cifar import vgg11_bn_quan, vgg16
from models.vanilla_models.vanilla_resnet_imagenet import resnet18
from models.quan_alexnet_imagenet import alexnet_quan
from models.quan_mobilenet_imagenet import mobilenet_v2_quan
from models.googlenet_cifar100 import googlenet
from models.quan_googlenet_cifar100 import googlenet_quan
from models.quan_densenet_cifar100 import densenet121_quan
from models.quan_shufflenetv2_cifar100 import shufflenetv2_quan
from models.quan_mobilenetv2_cifar100 import mobilenetv2_quan
from models.quan_squeezenet_cifar100 import squeezenet_quan
from models.quan_wideresnet import wideresnet_quan

# 디바이스 설정 (CUDA 사용 가능 여부 확인)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

use_pretrained = False # 사전 학습된 모델 사용 여부
model_name = 'wideresnet_quan' # 사용할 모델 이름 ########~~~ 여기도 수정해야됨 
save_path = './save/2025-07-03'
model_path = ''

# CIFAR-10 클래스 레이블 (10개의 클래스로 예시)
cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 클래스 개수 설정 (필요에 따라 CIFAR-10, CIFAR-100, ImageNet 등으로 조정 가능)
num_classes = 10  # CIFAR-10의 경우 10개, CIFAR-100의 경우 100, ImageNet의 경우 1000 등으로 설정 가능

# 이미지 크기 설정 # cifar10 -> 32 x 32 / imagenet... vgg, alexnet -> 224 x 224
#image_size = (224, 224)
image_size = (32, 32)

if model_name == 'resnet20_quan': 
    model = resnet20_quan(num_classes=num_classes)
    model_path = 'cifar10_resnet20_quan_160_binarized'
elif model_name == 'vgg11_quan': 
    model = vgg11_bn_quan(num_classes=num_classes)
    model_path = 'cifar10_vgg11_bn_quan_160_SGD_binarized'
elif model_name == 'wideresnet_quan':
    model = wideresnet_quan(depth=28, num_classes=num_classes)
    model_path = 'cifar10_wideresnet_quan_160_binarized'
else: 
    raise ValueError("지원되지 않는 모델 이름입니다.")

model = model.to(device)
model.eval()

# 사전 학습된 ResNet20 가중치 불러오기 !!!!!!!!!!!!!!!!!!! 여기 수정
checkpoint_path = save_path + '/' + model_path + '/model_best.pth.tar'
print(checkpoint_path)

# 사전 학습된 모델 로드
if not use_pretrained:
    # 사전 학습되지 않은 모델의 checkpoint 경로를 설정하고 불러옴 
    #checkpoint_path = '/path/to/checkpoint.pth' 
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False) 
    if 'state_dict' in checkpoint: 
        model.load_state_dict(checkpoint['state_dict']) 
        print("heyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
    else:
        model.load_state_dict(checkpoint) 

print(f"{model_name} model loaded.")




# 균일한 타겟 분포 생성 (num_classes에 따라 자동으로 분포 생성)
def get_uniform_target_distribution(num_classes):
    return torch.full((1, num_classes), 1.0 / num_classes).to(device)

'''
# 랜덤 이미지 생성 함수 (노이즈 기반)
def generate_random_image(batch_size=1, channels=3, height=image_size[0], width=image_size[1]):
    return torch.rand((batch_size, channels, height, width), device=device)
'''


# 랜덤 초기화를 다양화하는 함수
def generate_random_image(batch_size=1, channels=3, height=image_size[0], width=image_size[1]):
    if np.random.rand() > 0.5:
        # 완전 랜덤 노이즈
        return torch.rand((batch_size, channels, height, width), device=device)
    else:
        # Gaussian 노이즈 추가
        base_image = torch.rand((batch_size, channels, height, width), device=device)
        gaussian_noise = torch.randn((batch_size, channels, height, width), device=device) * 0.1
        return torch.clamp(base_image + gaussian_noise, min=0, max=1)


# PGD 기반 이미지 업데이트 함수
def update_image_to_uniform(model, images, num_classes, max_iters=1000, step_size=0.05, tol=1e-5, decay=0.99):
    target_dist = get_uniform_target_distribution(num_classes)  # 목표 균일 분포
    step_size = step_size

    for iter_num in range(max_iters):
        images.requires_grad = True  # 기울기 계산을 위한 설정

        # 모델 예측 및 확률 계산
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)

        # KL Divergence 손실 계산
        #kl_loss = F.kl_div(probs.log(), target_dist, reduction='batchmean')
        kl_loss = F.kl_div(F.log_softmax(outputs, dim=1), target_dist, reduction='batchmean')

        # 예측 분포와 균등 분포 사이의 L2 손실 추가
        l2_loss = torch.norm(probs - target_dist, p=2)


        # 엔트로피 최대화 손실 (예측이 고르게 분포되도록) ##젤 마지막에 추가 11.20
        #entropy_loss = -torch.sum(probs * torch.log(probs + 1e-12), dim=1).mean()

	# 클래스 간 분포 균일성 손실 (분산이 낮을수록 페널티)
        uniformity_loss = -torch.var(probs, dim=1).mean()

        # 총 손실 계산
        loss = kl_loss + 0.1 * l2_loss - 0.1 * uniformity_loss

        # 손실이 충분히 작아지면 멈춤
        if loss.item() < tol:
            break

        # 손실 기울기 계산
        model.zero_grad()
        loss.backward()

        # 이미지를 손실이 증가하는 방향으로 업데이트
        images = images.detach() - step_size * images.grad.sign()
        images = torch.clamp(images, min=0, max=1)  # 이미지의 값을 [0, 1]로 자름

        # 학습률 감소
        step_size *= decay

        # 100번마다 손실 값 출력
        if iter_num % 100 == 0:
            print(f"Iteration {iter_num}, Loss: {loss.item()}")

    return images

# 이미지 저장 함수
def save_image(tensor, filename):
    tensor = tensor.squeeze(0).detach().cpu()  # 배치 차원 제거 및 CPU로 이동
    image = transforms.ToPILImage()(tensor)  # 텐서를 PIL 이미지로 변환
    image.save(filename)

# 저장 디렉토리 설정
save_dir = "./data/challenge/"+ model_name
os.makedirs(save_dir, exist_ok=True)

# ===== 이미 존재하는 파일 중에서 가장 큰 번호 찾기 & 총 개수 확인 =====
existing_files = os.listdir(save_dir)
pattern = re.compile(r"challenge_image_(\d+)\.png")

max_index = 0
for fname in existing_files:
    match = pattern.match(fname)
    if match:
        num = int(match.group(1))
        if num > max_index:
            max_index = num

# 현재 디렉토리 내 이미지 개수
existing_count = len([f for f in existing_files if pattern.match(f)])

# 총 1000개 제한
remaining_count = max(0, 1000 - existing_count)
if remaining_count == 0:
    print("이미 1000개의 이미지가 존재합니다. 더 이상 생성하지 않습니다.")
else:
    start_index = max_index + 1
    print(f"{remaining_count}개의 이미지를 추가 생성합니다.")

    # ===== 랜덤 노이즈 기반 이미지 생성 및 저장 =====
    for i in range(remaining_count):
        random_image = generate_random_image()
        adv_image = update_image_to_uniform(model, random_image, num_classes=num_classes)

        save_path = os.path.join(save_dir, f'challenge_image_{start_index + i}.png')
        save_image(adv_image, save_path)
        print(f"Saved: {save_path}")

print("Adversarial images generation process finished.")

# 생성된 챌린지 이미지를 다시 불러와 모델에 넣고 예측 결과 출력 (Top 10 클래스)
def load_and_predict_challenge_images(model, challenge_dir, num_classes):
    transform = transforms.Compose([transforms.ToTensor()])
    challenge_images = []
    image_paths = [os.path.join(challenge_dir, img) for img in os.listdir(challenge_dir) if img.endswith('.png')]
    
    for img_path in image_paths:
        img = Image.open(img_path)
        img_tensor = transform(img).unsqueeze(0)  # 배치 차원 추가
        challenge_images.append(img_tensor)
    
    challenge_data = torch.cat(challenge_images, dim=0).to(device)

    # 모델에 입력하여 예측
    with torch.no_grad():
        outputs = model(challenge_data)
        probs = F.softmax(outputs, dim=1)

    # Top 10 예측 결과 출력
    topk = 10
    _, topk_indices = torch.topk(probs, topk, dim=1)
    topk_probs = probs.gather(1, topk_indices)

    # 각 이미지에 대해 상위 10개 클래스와 확률 출력
    for i in range(challenge_data.size(0)):
        print(f"Predictions for Challenge Image {i + 1}:")
        for j in range(topk):
            class_idx = topk_indices[i, j].item()
            class_prob = topk_probs[i, j].item() * 100
            print(f"Top {j + 1} Prediction: Class {class_idx} with confidence {class_prob:.2f}%")
        print()

# 챌린지 이미지에 대한 예측 수행
load_and_predict_challenge_images(model, save_dir, num_classes)


import os
import re

save_dir = "./data/challenge/" + model_name
os.makedirs(save_dir, exist_ok=True)

# ===== 이미 존재하는 파일 중에서 가장 큰 번호 찾기 & 총 개수 확인 =====
existing_files = os.listdir(save_dir)
pattern = re.compile(r"challenge_image_(\d+)\.png")





