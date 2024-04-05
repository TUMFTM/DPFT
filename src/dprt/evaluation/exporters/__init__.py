from dprt.evaluation.exporters.kradar import build_kradar


def build(name: str, *args, **kwargs):
    if name == 'kradar':
        return build_kradar(*args, **kwargs)
