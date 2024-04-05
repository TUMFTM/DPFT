from dprt.models.backbones.resnet import build_resnet
from dprt.models.backbones.regnet import build_regnet
from dprt.models.backbones.convnext import build_convnext
from dprt.models.backbones.swin import build_swin


def build_backbone(name: str, *args, **kwargs):
    if 'resnet' in name.lower():
        return build_resnet(*args, **kwargs)
    if 'regnet' in name.lower():
        return build_regnet(*args, **kwargs)
    if 'convnext' in name.lower():
        return build_convnext(*args, **kwargs)
    if 'swin' in name.lower():
        return build_swin(*args, **kwargs)
