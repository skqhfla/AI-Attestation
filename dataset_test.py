import torchvision.datasets as datasets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

# CIFAR-100 데이터 로드
data_path = 'home/ubuntu/BFA/BFA/data/cifar100'  # 데이터 경로 수정
transform = transforms.ToTensor()
train_dataset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)

# 이미지 샘플 확인
image, label = train_dataset[0]
plt.imshow(image.permute(1, 2, 0))  # 채널 순서 변환 (C, H, W) -> (H, W, C)
plt.title(f'Label: {label}')
plt.show()




