import pickle

from collections.abc import Iterable, Mapping, Sequence, Sized
from copy import deepcopy
from itertools import zip_longest

import torch


def _non_zipping_check(batch_data: Mapping | Iterable, detach: bool, pad: bool, fill_value):
    """Determines the lergest batch size of the given batch data.

    Utility function based on `decollate_batch`, to identify the largest batch size
    from the collated data.

    See `decollate_batch` for more details.

    Reference: MONAI (https://github.com/Project-MONAI/MONAI)

    Arguments:
        batch_data: Mapping or iterable of batched data.
        detach: Whether to detach the data.
        pad: Wheter to use padding.
        fill_value: Fill value if padding is used.

    Returns:
        batch_size: Largest batch size of the given batch data.
        non_iterable: List of non-iterable items.
        _deco: Dictionary or list with decollated items.
    """
    _deco: Mapping | Sequence
    if isinstance(batch_data, Mapping):
        _deco = {
            key: decollate_batch(batch_data[key], detach, pad=pad, fill_value=fill_value)
            for key in batch_data
        }

    elif isinstance(batch_data, Iterable):
        _deco = [decollate_batch(b, detach, pad=pad, fill_value=fill_value) for b in batch_data]

    else:
        raise NotImplementedError(f"Unable to de-collate: {batch_data}, type: {type(batch_data)}.")

    batch_size, non_iterable = 0, []

    for k, v in _deco.items() if isinstance(_deco, Mapping) else enumerate(_deco):
        if not isinstance(v, Iterable) or \
           isinstance(v, (str, bytes)) or \
           (isinstance(v, torch.Tensor) and v.ndim == 0):
            non_iterable.append(k)

        elif isinstance(v, Sized):
            batch_size = max(batch_size, len(v))

    return batch_size, non_iterable, _deco


def decollate_batch(batch, detach: bool = True, pad=True, fill_value=None):
    """De-collate a batch of data (for example, as produced by a `DataLoader`).

    Returns a list of structures with the original tensor's 0-th dimension sliced
    into elements using `torch.unbind`.

    Images originally stored as (B,C,H,W,[D]) will be returned as (C,H,W,[D]).
    Other information, such as metadata, may have been stored in a list
    (or a list inside nested dictionaries). In this case we return the element
    of the list corresponding to the batch idx.

    Return types aren't guaranteed to be the same as the original, since
    numpy arrays will have been converted to torch.Tensor, sequences may
    be converted to lists of tensors, mappings may be converted into dictionaries.

    For example:

    .. code-block:: python

        batch_data = {
            "image": torch.rand((2,1,10,10)),
            DictPostFix.meta("image"): {"scl_slope": torch.Tensor([0.0, 0.0])}
        }
        out = decollate_batch(batch_data)
        print(len(out))
        >>> 2

        print(out[0])
        >>> {'image': tensor([[[4.3549e-01...43e-01]]]),
             DictPostFix.meta("image"): {'scl_slope': 0.0}}

        batch_data = [torch.rand((2,1,10,10)), torch.rand((2,3,5,5))]
        out = decollate_batch(batch_data)
        print(out[0])
        >>> [tensor([[[4.3549e-01...43e-01]]], tensor([[[5.3435e-01...45e-01]]])]

        batch_data = torch.rand((2,1,10,10))
        out = decollate_batch(batch_data)
        print(out[0])
        >>> tensor([[[4.3549e-01...43e-01]]])

        batch_data = {
            "image": [1, 2, 3], "meta": [4, 5],  # undetermined batch size
        }
        out = decollate_batch(batch_data, pad=True, fill_value=0)
        print(out)
        >>> [{'image': 1, 'meta': 4}, {'image': 2, 'meta': 5}, {'image': 3, 'meta': 0}]
        out = decollate_batch(batch_data, pad=False)
        print(out)
        >>> [{'image': 1, 'meta': 4}, {'image': 2, 'meta': 5}]

    Reference: MONAI (https://github.com/Project-MONAI/MONAI)

    Arguments:
        batch: data to be de-collated.
        detach: whether to detach the tensors. Scalars tensors will be detached into number types
            instead of torch tensors.
        pad: when the items in a batch indicate different batch size, whether to pad all the
            sequences to the longest. If False, the batch size will be the length of the
            shortest sequence.
        fill_value: when `pad` is True, the `fillvalue` to use when padding, defaults to `None`.

    Returns:
        De-collated list of batch items.
    """
    if batch is None:
        return batch
    if isinstance(batch, (float, int, str, bytes)) or (
        type(batch).__module__ == "numpy" and not isinstance(batch, Iterable)
    ):
        return batch
    if isinstance(batch, torch.Tensor):
        if detach:
            batch = batch.detach()
        if batch.ndim == 0:
            return batch.item() if detach else batch
        out_list = torch.unbind(batch, dim=0)

        if out_list[0].ndim == 0 and detach:
            return [t.item() for t in out_list]
        return list(out_list)

    b, non_iterable, deco = _non_zipping_check(batch, detach, pad, fill_value)
    if b <= 0:  # all non-iterable, single item "batch"? {"image": 1, "label": 1}
        return deco
    if pad:  # duplicate non-iterable items to the longest batch
        for k in non_iterable:
            deco[k] = [deepcopy(deco[k]) for _ in range(b)]
    if isinstance(deco, Mapping):
        _gen = zip_longest(*deco.values(), fillvalue=fill_value) if pad else zip(*deco.values())
        ret = [dict(zip(deco, item)) for item in _gen]
        return pickle_operations(ret, is_encode=False)
    if isinstance(deco, Iterable):
        _gen = zip_longest(*deco, fillvalue=fill_value) if pad else zip(*deco)
        ret_list = [list(item) for item in _gen]
        return pickle_operations(ret_list, is_encode=False)
    raise NotImplementedError(f"Unable to de-collate: {batch}, type: {type(batch)}.")


def pickle_operations(data, key: str = "_transforms", is_encode: bool = True):
    """
    Applied_operations are dictionaries with varying sizes, this method
    converts them to bytes so that we can (de-)collate.

    Args:
        data: a list or dictionary with substructures to be pickled/unpickled.
        key: the key suffix for the target substructures, defaults to "_transforms".
        is_encode: whether it's encoding using pickle.dumps (True) or
            decoding using pickle.loads (False).
    """
    if isinstance(data, Mapping):
        data = dict(data)
        for k in data:
            if f"{k}".endswith(key):
                if is_encode and not isinstance(data[k], bytes):
                    data[k] = pickle.dumps(data[k], 0)
                if not is_encode and isinstance(data[k], bytes):
                    data[k] = pickle.loads(data[k])
        return {k: pickle_operations(v, key=key, is_encode=is_encode) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [pickle_operations(item, key=key, is_encode=is_encode) for item in data]
    return data
