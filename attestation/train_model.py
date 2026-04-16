import os
import timm
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from tqdm import tqdm
from timm.data import Mixup
from timm.utils import ModelEma
from torch.utils.data import DataLoader
from timm.loss import SoftTargetCrossEntropy
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights

from libs import create_directory
from libs import quan_Conv2d, quan_Linear

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CHKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')

MODEL_NAME = 'TResNet'.lower()  # EfficientNetv2, ViT, TResNet


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


def train(model, loader, optimizer, criterion, mixup_fn, ema, device):
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    for inputs, targets in tqdm(loader, total=len(loader)):
        inputs, targets = inputs.to(device), targets.to(device)

        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if ema:
            ema.update(model)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        hard_targets = torch.argmax(targets, dim=1) if targets.ndim == 2 else targets
        correct += predicted.eq(hard_targets).sum().item()

    acc = 100. * correct / total
    loss = train_loss / total
    return loss, acc


def evaluate(model, loader, criterion, device):
    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, total=len(loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            hard_targets = torch.argmax(targets, dim=1) if targets.ndim == 2 else targets
            correct += predicted.eq(hard_targets).sum().item()

    acc = 100. * correct / total
    loss = test_loss / total
    return loss, acc


def load_and_test_model(loader, creiterion, device):
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
    model.to(device)

    weights = torch.load(os.path.join(CHKPT_DIR, f'{MODEL_NAME}_cifar100_best_weights.pth'),
                         map_location=device,
                         weights_only=True)
    model.load_state_dict(weights)

    loss, acc = evaluate(model, loader, creiterion, device)
    print(f'[{MODEL_NAME}] Loss: {loss}, Acc: {acc}')
    

def main():
    input_size = 224
    batch_size = 32 if MODEL_NAME == 'efficientnetv2' else 64  # EfficientNetv2: 32, ViT: 64, TResNet: 64
    epochs = 100
    lr = 0.001
    weight_decay = 5e-4
    ema_decay = 0.9999
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform_train = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomCrop(input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        ),
        transforms.RandomErasing(p=0.25)
    ])

    transform_test = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        )
    ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform_test
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    if MODEL_NAME == 'efficientnetv2':
        weights = EfficientNet_V2_L_Weights.DEFAULT
        model = efficientnet_v2_l(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.25, inplace=True),
            nn.Linear(model.classifier[-1].in_features, 100),
        )
    elif MODEL_NAME == 'vit':
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, 100)
    elif MODEL_NAME == 'tresnet':
        model = timm.create_model('tresnet_l', pretrained=True, num_classes=100)
    else:
        raise ValueError(f'Invalid model name: {MODEL_NAME}')
    
    quantize(model)
    model.to(device)

    ema = ModelEma(model, decay=ema_decay, device=device)
    mixup_fn = Mixup(
        mixup_alpha=0.2,
        cutmix_alpha=0.2,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=0.1,
        num_classes=100
    )

    train_criterion = SoftTargetCrossEntropy()
    test_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0

    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, train_criterion, mixup_fn, ema, device)
        test_loss, test_acc = evaluate(ema.ema, test_loader, test_criterion, device)
        scheduler.step()

        print(f'[Epoch {epoch + 1}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(ema.ema.state_dict(), os.path.join(CHKPT_DIR, f'{MODEL_NAME}_cifar100_best_weights.pth'))
            print(f'Saved best model weights with accuracy: {test_acc:.2f}%')

    load_and_test_model(test_loader, test_criterion, device)


if __name__ == "__main__":
    create_directory(os.path.join(CHKPT_DIR))
    main()
