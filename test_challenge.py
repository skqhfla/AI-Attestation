import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
import models  # main.py와 동일한 경로에 있는 models 모듈 불러오기

# 디바이스 설정 (CUDA 사용 가능 여부 확인)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 클래스 개수 설정
num_classes = 10  # 필요에 따라 CIFAR-10, CIFAR-100, ImageNet에 맞게 설정

# 데이터 전처리 (CIFAR-10용, 다른 데이터셋에 맞춰 수정 가능)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# ResNet20 모델을 CIFAR-10에 맞춰 불러오기 (models 모듈 사용)
from models.quan_resnet_cifar import resnet20_quan
from models.quan_vgg_cifar import vgg11_quan
model = vgg11_quan(num_classes=num_classes)  # CIFAR-10은 10개의 클래스
model = model.to(device)
model.eval()

# 사전 학습된 ResNet20 가중치 불러오기
#checkpoint_path = '/home/ubuntu/BFA/BFA/save/2024-09-25/cifar10_resnet20_quan_160_SGD_binarized/model_best.pth.tar'
checkpoint_path = '/home/ubuntu/BFA/BFA/save/2024-10-17/cifar10_vgg11_quan_160_SGD_binarized/model_best.pth.tar'

checkpoint = torch.load(checkpoint_path, map_location=device)
if 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

print("ResNet20 model loaded.")

# 손실 함수 정의
criterion = nn.CrossEntropyLoss()

# Challenge 이미지 저장 디렉토리
challenge_dir = './data/challenge/'

"""
# Challenge 이미지 로드 함수 정의
def load_challenge_images(challenge_dir, transform):
    challenge_images = []
    image_paths = [os.path.join(challenge_dir, img) for img in os.listdir(challenge_dir) if img.endswith('.png')]
    for img_path in image_paths:
        img = Image.open(img_path)
        img_tensor = transform(img).unsqueeze(0)  # 배치 차원 추가
        challenge_images.append(img_tensor)
    return torch.cat(challenge_images, dim=0)  # 모든 이미지 텐서를 하나로 결합

# Challenge 이미지를 로드
challenge_data = load_challenge_images(challenge_dir, transform_test)
print(f"Loaded {challenge_data.size(0)} challenge images.")

# Challenge 이미지에 대한 모델 예측
with torch.no_grad():
    challenge_data = challenge_data.to(device)
    outputs = model(challenge_data)
    probs = torch.softmax(outputs, dim=1)

# 예측 결과에서 top 10 클래스 및 확률 출력
topk = 10
_, topk_indices = torch.topk(probs, topk, dim=1)
topk_probs = probs.gather(1, topk_indices)

# 각 이미지에 대해 예측된 상위 10개 클래스와 확률 출력
for i in range(challenge_data.size(0)):
    print(f"Predictions for Challenge Image {i + 1}:")
    for j in range(topk):
        class_idx = topk_indices[i, j].item()
        class_prob = topk_probs[i, j].item() * 100
        print(f"Top {j + 1} Prediction: Class {class_idx} with confidence {class_prob:.2f}%")
    print()
"""





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
load_and_predict_challenge_images(model, challenge_dir, num_classes)














