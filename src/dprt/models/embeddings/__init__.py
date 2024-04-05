from torch import nn

from dprt.models.embeddings.sinusoidal import build_sinusoidal_embedding


def build_embedding(name: str, *args, **kwargs) -> nn.Module:
    if 'sinusoidal' in name:
        return build_sinusoidal_embedding(*args, **kwargs)
