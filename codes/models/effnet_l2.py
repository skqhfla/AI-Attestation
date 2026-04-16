# models/effnet_l2.py

import timm
import torch.nn as nn
from .quantization import *

def effnet_l2(num_classes=100):
    model = timm.create_model("efficientnet_l2", pretrained=False)
    model.classifier = quan_Linear(model.classifier.in_features, num_classes)
    return model

