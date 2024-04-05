from dprt.evaluation.evaluator import build_evaluator


def evaluate(*args, **kwargs):
    return build_evaluator(*args, **kwargs)
