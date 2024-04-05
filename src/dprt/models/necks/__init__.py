from dprt.models.necks.fpn import build_fpn


def build_neck(name: str, *args, **kwargs):
    if 'fpn' in name.lower():
        return build_fpn(name, *args, **kwargs)
