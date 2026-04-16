import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_l
import timm
import hashlib
import time
# 모델을 불러오기 위한 models 모듈 사용 (main.py와 동일 경로에 있다고 가정)
import models  # main.py와 동일한 경로에 있는 models 모듈 불러오기
from models.quantization import quan_Conv2d, quan_Linear

# ResNet20 모델을 CIFAR-10에 맞춰 불러오기 (models 모듈 사용) !!!!!!!!!!!!!!!!!!!! 수정 
from models.quan_resnet_cifar import resnet20_quan
from models.quan_vgg_cifar import vgg11_bn_quan, vgg16
from models.quan_wideresnet import wideresnet_quan

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


def hash_model_weights(model, checkpoint_path):
    """
    Calculates the SHA256 hash of the weights from a PyTorch model's state_dict.

    Parameters:
        model (torch.nn.Module): The PyTorch model instance.
        checkpoint_path (str): Path to the saved checkpoint file.

    Returns:
        dict: Information including model name, parameter size, and hash value.
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract state_dict
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Serialize weights to a byte stream
    sha256_hash = hashlib.sha256()
    parameter_size_bytes = 0

    times = []
    for _ in range(1000):
        start_time = time.perf_counter()
        for key, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                data_bytes = param.cpu().numpy().tobytes()
                sha256_hash.update(data_bytes)
                parameter_size_bytes += len(data_bytes)

        end_time = time.perf_counter()
        times.append((end_time - start_time)*1000)

    avg_time = sum(times) / len(times)

    hash_value = sha256_hash.hexdigest()
    hashing_time_ms = (end_time - start_time) * 1000  # Convert to ms

    return {
        "Model Name": model.__class__.__name__,
        "Parameter Size (MBytes)": parameter_size_bytes / (1024 * 1024),
        "Avg Hasing Time (ms)": avg_time,
        "Hashing Time (ms)": hashing_time_ms,
        "SHA256 Hash": hash_value
    }

def inference_time(model):
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32)

    for _ in range(10):
        _ = model(dummy_input)

    times = []
    for _ in range(1000):
        start = time.time()
        _ = model(dummy_input)
        end = time.time()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    print(f"Average Inference Time: {avg_time*1000:.3f} ms")

# Example usage
if __name__ == "__main__":
    
    # Model setup
#    checkpoint_path = '/home/cpss/Attestation/AIattestation/codebackup/codes/save/2025-08-19/cifar10_resnet20_quan_160_binarized/model_best.pth.tar'
    checkpoint_path = '/home/cpss/Attestation/AIattestation/codebackup/codes/save/2025-08-29/efficientnetv2_cifar100_best_weights.pth'
    #model = resnet20_quan(num_classes=10)  # Replace with your model class
    model = efficientnet_v2_l()
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.25, inplace=True),
        nn.Linear(model.classifier[-1].in_features, 100),
    )
    '''
    model = timm.create_model('tresnet_l', pretrained=True, num_classes=100)
    '''
    quantize(model)

    # Calculate hash
    results = hash_model_weights(model, checkpoint_path)

    # Print results
    print("Model Weight Hashing Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

    inference_time(model)




