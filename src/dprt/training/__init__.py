from dprt.training.trainer import build_trainer


def train(*args, **kwargs):
    return build_trainer(*args, **kwargs)
