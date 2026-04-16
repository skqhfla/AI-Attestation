import timm
import torch.nn as nn
from .quantization import quan_Conv2d, quan_Linear

def _replace_layers_with_quantization(module, w_bit=4, a_bit=4):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            q = quan_Conv2d(
                child.in_channels, child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=(child.bias is not None),
            )
            setattr(module, name, q)
        elif isinstance(child, nn.Linear):
            q = quan_Linear(
                child.in_features, child.out_features,
                bias=(child.bias is not None),
            )
            setattr(module, name, q)
        else:
            _replace_layers_with_quantization(child, w_bit=w_bit, a_bit=a_bit)

def vit_quan(num_classes=100):
    # timm에서 ViT-B/16 (ImageNet-21K pretrain) 로드
    print(f"vit_quan class {num_classes}")
    model = timm.create_model('vit_base_patch16_224_in21k',
                              pretrained=True,
                              num_classes=num_classes)
    # 계층 교체
    _replace_layers_with_quantization(model, w_bit=4, a_bit=4)
    return model
    
def vit(num_classes=100):
    # timm에서 ViT-B/16 (ImageNet-21K pretrain) 로드
    print(f"vit class {num_classes}")
    model = timm.create_model('vit_base_patch16_224_in21k',
                              pretrained=True)
    # 계층 교체
   # _replace_layers_with_quantization(model, w_bit=4, a_bit=4)
    return model

