import os

from typing import Tuple

import torch

from dprt.models.dprt import build_dprt


def build(model: str, *args, **kwargs):
    if model == 'dprt':
        return build_dprt(*args, **kwargs)


def load(checkpoint: str, *args, **kwargs) -> Tuple[torch.nn.Module, int, str]:
    filename = os.path.splitext(os.path.basename(checkpoint))[0]
    timestamp, _, epoch = filename.split('_')
    return torch.load(checkpoint), int(epoch), timestamp
