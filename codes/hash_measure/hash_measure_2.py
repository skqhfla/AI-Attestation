import torch
import hashlib
import time

# 모델을 불러오기 위한 models 모듈 사용 (main.py와 동일 경로에 있다고 가정)
import models  # main.py와 동일한 경로에 있는 models 모듈 불러오기

# ResNet20 모델을 CIFAR-10에 맞춰 불러오기 (models 모듈 사용) !!!!!!!!!!!!!!!!!!!! 수정 
from models.quan_resnet_cifar import resnet20_quan
from models.quan_resnet_imagenet import resnet18_quan
from models.quan_vgg_cifar import vgg11_quan, vgg16
from models.vanilla_models.vanilla_resnet_imagenet import resnet18
from models.quan_alexnet_imagenet import alexnet_quan
from models.quan_mobilenet_imagenet import mobilenet_v2_quan
from models.googlenet_cifar100 import googlenet
from models.quan_googlenet_cifar100 import googlenet_quan
from models.quan_densenet_cifar100 import densenet121_quan
from models.quan_shufflenetv2_cifar100 import shufflenetv2_quan
from models.quan_mobilenetv2_cifar100 import mobilenetv2_quan
from models.quan_squeezenet_cifar100 import squeezenet_quan


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
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint  # Handle direct state_dict saving

    # Serialize weights to a byte stream
    parameter_data = b""
    for key, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            parameter_data += param.cpu().numpy().tobytes()  # Convert to bytes

    # Calculate parameter size
    parameter_size_bytes = len(parameter_data)

    # Compute SHA256 hash
    sha256_hash = hashlib.sha256()
    start_time = time.time()
    sha256_hash.update(parameter_data)
    end_time = time.time()

    hash_value = sha256_hash.hexdigest()
    hashing_time_ms = (end_time - start_time) * 1000  # Convert to ms

    return {
        "Model Name": model.__class__.__name__,
        "Parameter Size (MB)": parameter_size_bytes / (1024 * 1024),
        "Hashing Time (ms)": hashing_time_ms,
        "SHA256 Hash": hash_value
    }

# Example usage
if __name__ == "__main__":
    # 모델 설정
    model_name = "vgg11_quan"
    checkpoint_path = '/home/ubuntu/BFA/BFA/save/2024-11-08/cifar100_densenet121_quan_160_SGD_binarized/model_best.pth.tar'

    # Load the model structure
    model = densenet121_quan(num_classes=100)  # Replace with your model class
    model = alexnet_quan(pretrained=True, num_classes=1000)

    # Calculate hash
    results = hash_model_weights(model, checkpoint_path)

    # Print results
    print("Model Weight Hashing Results:") 
    for key, value in results.items():
        print(f"{key}: {value}")




