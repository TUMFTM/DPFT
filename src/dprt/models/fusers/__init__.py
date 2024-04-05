from dprt.models.fusers.mpfusion import build_mpfusion


def build_fuser(name: str, *args, **kwargs):
    if 'impfusion' in name.lower():
        return build_mpfusion(*args, **kwargs)
