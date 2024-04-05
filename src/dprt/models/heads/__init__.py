from dprt.models.heads.detection import build_detection_head


def build_head(name: str, *args, **kwargs):
    if 'detection' in name.lower():
        return build_detection_head(name, *args, **kwargs)
