from dprt.models.queries.data_agnostic import build_data_agnostic_query
from dprt.models.queries.learnable import build_learnable_query


def build_querent(name: str, *args, **kwargs):
    if 'data_agnostic' in name.lower():
        return build_data_agnostic_query(name, *args, **kwargs)
    if 'learnable' in name.lower():
        return build_learnable_query(name, *args, **kwargs)
