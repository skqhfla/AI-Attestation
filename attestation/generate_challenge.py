import os
import re
import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from torchvision.models import vit_b_16
from torchvision.models import efficientnet_v2_l

from libs import create_directory
from libs import quan_Conv2d, quan_Linear

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CHKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
CHALL_DIR = os.path.join(BASE_DIR, 'challenges')

MODEL_NAME = 'efficientnetv2'.lower()  # EfficientNetv2, ViT, TResNet
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 100


def quantize(model):
    for name, child in model.named_children():
        if isinstance(child, nn.Conv2d):
            quan_layer = quan_Conv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None
            )
            quan_layer.weight.data = child.weight.data.clone()
            if child.bias is not None:
                quan_layer.bias.data = child.bias.data.clone()
            setattr(model, name, quan_layer)
        elif isinstance(child, nn.Linear):
            quan_layer = quan_Linear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None
            )
            quan_layer.weight.data = child.weight.data.clone()
            if child.bias is not None:
                quan_layer.bias.data = child.bias.data.clone()
            setattr(model, name, quan_layer)
        else:
            quantize(child)


def test_model(model):
    import torchvision

    from tqdm import tqdm
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        )
    ])
    dataset = torchvision.datasets.CIFAR100(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, total=len(loader)):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            hard_targets = torch.argmax(targets, dim=1) if targets.ndim == 2 else targets
            correct += predicted.eq(hard_targets).sum().item()

    acc = 100. * correct / total
    loss = test_loss / total
    print(f'[{MODEL_NAME}] Loss: {loss}, Acc: {acc}')


def get_uniform_target_distribution(num_classes):
    return torch.full((1, num_classes), 1.0 / num_classes).to(DEVICE)

def generate_random_image(batch_size=1, channels=3, height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]):
        return torch.rand((batch_size, channels, height, width), device=DEVICE)
'''
def generate_random_image(batch_size=1, channels=3, height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]):
    if np.random.rand() > 0.5:
        return torch.rand((batch_size, channels, height, width), device=DEVICE)
    else:
        base_image = torch.rand((batch_size, channels, height, width), device=DEVICE)
        gaussian_noise = torch.randn((batch_size, channels, height, width), device=DEVICE) * 0.1
        return torch.clamp(base_image + gaussian_noise, min=0, max=1)
'''

def update_image_to_uniform(model, images, num_classes, max_iters=1000, step_size=0.05, tol=1e-5, decay=0.99):
    target_dist = get_uniform_target_distribution(num_classes)
    step_size = step_size

    for iter_num in range(max_iters):
        images.requires_grad = True

        outputs = model(images)
        probs = F.softmax(outputs, dim=1)

        # kl_loss = F.kl_div(probs.log(), target_dist, reduction='batchmean')
        kl_loss = F.kl_div(F.log_softmax(outputs, dim=1), target_dist, reduction='batchmean')

        l2_loss = torch.norm(probs - target_dist, p=2)

        # entropy_loss = -torch.sum(probs * torch.log(probs + 1e-12), dim=1).mean()
        uniformity_loss = -torch.var(probs, dim=1).mean()

        loss = kl_loss + 0.1 * l2_loss - 0.1 * uniformity_loss

        if loss.item() < tol:
            break

        model.zero_grad()
        loss.backward()

        images = images.detach() - step_size * images.grad.sign()
        images = torch.clamp(images, min=0, max=1)

        step_size *= decay

        if iter_num % 100 == 0:
            print(f"Iteration {iter_num}, Loss: {loss.item()}")

    return images


def save_image(tensor, filename):
    tensor = tensor.squeeze(0).detach().cpu()
    image = transforms.ToPILImage()(tensor)
    image.save(filename)


def load_and_predict_challenge_images(model, challenge_dir, num_classes):
    transform = transforms.Compose([transforms.ToTensor()])
    challenge_images = []
    image_paths = [os.path.join(challenge_dir, img) for img in os.listdir(challenge_dir) if img.endswith('.png')]
    
    for img_path in image_paths:
        img = Image.open(img_path)
        img_tensor = transform(img).unsqueeze(0)
        challenge_images.append(img_tensor)
    
    challenge_data = torch.cat(challenge_images, dim=0).to(DEVICE)

    with torch.no_grad():
        outputs = model(challenge_data)
        probs = F.softmax(outputs, dim=1)

    topk = 10
    _, topk_indices = torch.topk(probs, topk, dim=1)
    topk_probs = probs.gather(1, topk_indices)

    for i in range(challenge_data.size(0)):
        print(f"Predictions for Challenge Image {i + 1}:")
        for j in range(topk):
            class_idx = topk_indices[i, j].item()
            class_prob = topk_probs[i, j].item() * 100
            print(f"Top {j + 1} Prediction: Class {class_idx} with confidence {class_prob:.2f}%")
        print()


def main():
    save_dir = os.path.join(CHALL_DIR, MODEL_NAME, "random")
    create_directory(save_dir)

    if MODEL_NAME == 'efficientnetv2':
        model = efficientnet_v2_l()
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.25, inplace=True),
            nn.Linear(model.classifier[-1].in_features, 100),
        )
    elif MODEL_NAME == 'vit':
        model = vit_b_16()
        model.heads.head = nn.Linear(model.heads.head.in_features, 100)
    elif MODEL_NAME == 'tresnet':
        model = timm.create_model('tresnet_l', pretrained=True, num_classes=100)
    else:
        raise ValueError(f'Invalid model name: {MODEL_NAME}')
    
    quantize(model)
    model.to(DEVICE)
    model.eval()

    weights = torch.load(os.path.join(CHKPT_DIR, f'{MODEL_NAME}_cifar100_best_weights.pth'),
                         map_location=DEVICE,
                         weights_only=True)
    model.load_state_dict(weights)

    test_model(model)

    existing_files = os.listdir(save_dir)
    pattern = re.compile(r"challenge_image_(\d+)\.png")

    max_index = 0
    for fname in existing_files:
        match = pattern.match(fname)
        if match:
            num = int(match.group(1))
            if num > max_index:
                max_index = num
    
    existing_count = len([f for f in existing_files if pattern.match(f)])

    remaining_count = max(0, 1000 - existing_count)
    if remaining_count == 0:
        print("이미 1000개의 이미지가 존재합니다. 더 이상 생성하지 않습니다.")
    else:
        start_index = max_index + 1
        print(f"{remaining_count}개의 이미지를 추가 생성합니다.")

        for i in range(remaining_count):
            random_image = generate_random_image()
    #        adv_image = update_image_to_uniform(model, random_image, num_classes=NUM_CLASSES)

            save_path = os.path.join(save_dir, f'challenge_image_{start_index + i}.png')
            #save_image(adv_image, save_path)
            save_image(random_image, save_path)
            print(f"Saved: {save_path}")

    print("Adversarial images generation process finished.")

    load_and_predict_challenge_images(model, save_dir, NUM_CLASSES)


if __name__ == "__main__":
    main()
