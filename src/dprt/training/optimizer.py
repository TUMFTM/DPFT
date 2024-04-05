from functools import partial

import torch


def build_optimizer(name: str, *args, **kwargs):
    return partial(getattr(torch.optim, name), *args, **kwargs)
