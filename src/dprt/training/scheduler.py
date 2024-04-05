from functools import partial
from typing import Any, Dict, List

import torch


def chained_scheduler(optimizer, schedulers: List[Dict[str, Any]], *args, **kwargs):
    # Initialize schedulers
    schedulers = [
        getattr(torch.optim.lr_scheduler, scheduler.pop('name'))(optimizer, **scheduler)
        for scheduler in schedulers
    ]

    return torch.optim.lr_scheduler.ChainedScheduler(schedulers, *args, **kwargs)


def sequential_lr(optimizer, schedulers: List[Dict[str, Any]], *args, **kwargs):
    # Initialize schedulers
    schedulers = [
        getattr(torch.optim.lr_scheduler, scheduler.pop('name'))(optimizer, **scheduler)
        for scheduler in schedulers
    ]

    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers, *args, **kwargs)


def build_scheduler(name: str, *args, **kwargs):
    if name == "ChainedScheduler":
        return partial(chained_scheduler, *args, **kwargs)
    if name == "SequentialLR":
        return partial(sequential_lr, *args, **kwargs)

    return partial(getattr(torch.optim.lr_scheduler, name), *args, **kwargs)
