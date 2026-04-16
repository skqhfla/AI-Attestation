import torch
import torch.nn as nn
from timm.models import create_model
from .quantization import quan_Linear
from .ml_decoder.ml_decoder import MLDecoder, TransformerDecoderLayerOptimal

# Patch TransformerDecoderLayerOptimal to add self_attn attribute for compatibility
if hasattr(TransformerDecoderLayerOptimal, '__init__'):
    orig_init = TransformerDecoderLayerOptimal.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.self_attn = self.multihead_attn

    TransformerDecoderLayerOptimal.__init__ = patched_init
    
if hasattr(TransformerDecoderLayerOptimal, 'forward'):
    orig_forward = TransformerDecoderLayerOptimal.forward

    def patched_forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                        tgt_key_padding_mask=None, memory_key_padding_mask=None,
                        tgt_is_causal=None):
        return orig_forward(self, tgt, memory, tgt_mask, memory_mask,
                            tgt_key_padding_mask, memory_key_padding_mask)

    TransformerDecoderLayerOptimal.forward = patched_forward

class swinmldecoder_quan(nn.Module):
    def __init__(self, num_classes=100, decoder_embedding=768, decoder_layers=1):
        super().__init__()

        self.backbone = create_model('swin_tiny_patch4_window7_224', pretrained=True)
        self.backbone.head = nn.Identity()  # Remove classification head

        if hasattr(self.backbone, 'forward_features'):
            self._use_forward_features = True
        else:
            self._use_forward_features = False

        self.decoder = MLDecoder(
            num_classes=num_classes,
            initial_num_features=768,  # fixed to match actual Swin output
            decoder_embedding=decoder_embedding,
            num_of_groups=decoder_embedding // 64,
        )

    def forward(self, x):
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        feats = self.backbone.forward_features(x) if self._use_forward_features else self.backbone(x)

        if feats.ndim == 4:
            feats = feats.view(feats.size(0), -1, feats.size(-1))  # [B, 7, 7, 768] → [B, 49, 768]
        elif feats.ndim == 3:
            feats = feats  # already in [B, N, 768]
        else:
            raise ValueError(f"Unexpected feature shape: {feats.shape}")

        out = self.decoder(feats)
        return out

