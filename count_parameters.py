import torch.nn as nn
import models as my_models

import argparse

model_names = sorted(name for name in my_models.__dict__ if name.islower() and not name.startswith("__") and callable(my_models.__dict__[name]))

parser = argparse.ArgumentParser(description="please put on model name")
parser.add_argument("--arch", type=str, choices=model_names, required=True)

args = parser.parse_args()

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total' : total_params, 'Trainable' : trainable_params}

if args.arch.startswith("wideresnet"):
    net = my_models.__dict__[args.arch](depth=28, num_classes=10)
else:
    net = my_models.__dict__[args.arch](num_classes=10)

params = count_parameters(net)
print(f"Total parameters: {params['Total']}")
print(f"Trainable parameters: {params['Trainable']}")
