import hashlib
import time
import os
import torch

def measure_model_hashing_time(model_path):
    """
    Measures the time it takes to calculate the SHA256 hash of a saved model's parameters.

    Parameters:
        model_path (str): Path to the saved model file.

    Returns:
        dict: A dictionary containing the model name, file size, parameter size, and hashing time.
    """
    # Model name
    model_name = os.path.basename(model_path)

    # Get model file size in MB
    model_size_bytes = os.path.getsize(model_path)
    model_size_mb = model_size_bytes / (1024 * 1024)

    # Load the model parameters
    model = torch.load(model_path, map_location='cpu')

    # Serialize parameters to a byte stream
    parameter_data = b""
    for param_tensor in model.values():
        if isinstance(param_tensor, torch.Tensor):
            parameter_data += param_tensor.cpu().numpy().tobytes()

    parameter_size_bytes = len(parameter_data)

    # Hash the parameters
    start_time = time.time()
    sha256_hash = hashlib.sha256()
    sha256_hash.update(parameter_data)
    hash_value = sha256_hash.hexdigest()
    end_time = time.time()

    # Calculate hashing time in milliseconds
    hashing_time_ms = (end_time - start_time) * 1000

    # Return results
    return {
        "Model Name": model_name,
        "Model File Size (MB)": model_size_mb,
        "Parameter Size (Bytes)": parameter_size_bytes,
        "Hashing Time (ms)": hashing_time_ms,
        "SHA256 Hash": hash_value
    }

# Example usage
model_path = "/home/ubuntu/BFA/BFA/save/2024-11-05/cifar100_vgg16_160_SGD_binarized/checkpoint.pth.tar"  # Replace with the path to your saved model file
result = measure_model_hashing_time(model_path)
print("Model Hashing Results:")
for key, value in result.items():
    print(f"{key}: {value}")



