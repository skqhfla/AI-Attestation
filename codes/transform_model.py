import torch

# 원래 checkpoint 경로
ckpt_path = './save/2025-07-03/cifar10_wideresnet_quan_160_binarized/model_best.pth.tar'

# checkpoint 로드
ckpt = torch.load(ckpt_path, weights_only=False)
print(ckpt.keys())
model_state_dict = ckpt['state_dict']  # 또는 'model' 이라는 key일 수도 있으니 꼭 확인!
torch.save(model_state_dict, './save/2025-07-03/cifar10_wideresnet_quan_160_binarized/state_dict_only.pth')

