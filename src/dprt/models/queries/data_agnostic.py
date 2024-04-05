from __future__ import annotations  # noqa: F407

from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Union, Sequence

import torch

from torch import nn

from dprt.models.utils.transformations import build_transformation


class DataAgnosticStaticQueries(nn.Module):
    def __init__(self,
                 resolution: List[int] = None,
                 minimum: List[float] = None,
                 maximum: List[float] = None,
                 transformation: nn.Module = None,
                 distribution: Union[str, List[str]] = None,
                 **kwargs):
        """Returns reference query points of arbitrary dimensionality.

        Arguments:
            resolution: Number of query points to generate for each dimension.
            minimum: Minimum value of the smalles reference point in each dimension.
            maximum: Maximum value of the largest reference point in each dimension.
            transformation: Transformation (coordinate transformation) to apply
                to the query points.
            distribution: Distrbution to apply to the query points during
                initialization (default linear).
        """
        super().__init__()

        # Initialize instance attributes
        self.resolution = resolution if resolution is not None else []
        self.minimum = minimum if minimum is not None else []
        self.maximum = maximum if maximum is not None else []
        if transformation is not None:
            self.transformation = transformation
        else:
            self.transformation = nn.Identity()

        if distribution is None:
            self.distribution = ['linear'] * len(self.resolution)
        elif isinstance(distribution, (list, tuple)):
            self.distribution = distribution
        else:
            self.distribution = [distribution] * len(self.resolution)

        # Check input argument values
        assert len(self.resolution) == \
            len(self.minimum) == \
            len(self.maximum) == \
            len(self.distribution)

        # Get distribution function
        self._dist_fns = self._get_dist_fns(self.distribution)

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        transformation = build_transformation(config.get('transformation'))
        return cls(
            config['resolution'],
            config['minimum'],
            config['maximum'],
            transformation=transformation,
            distribution=config.get('distribution')
        )

    @staticmethod
    def _get_input_device(input):
        if isinstance(input, torch.Tensor):
            device = input.device
        elif isinstance(input, dict):
            device = input[list(input.keys())[0]].device
        elif isinstance(input, (list, tuple, set)):
            device = input[0].device
        return device

    @staticmethod
    def _get_input_dtype(input):
        if isinstance(input, torch.Tensor):
            dtype = input.dtype
        elif isinstance(input, dict):
            dtype = input[list(input.keys())[0]].dtype
        elif isinstance(input, (list, tuple, set)):
            dtype = input[0].dtype
        return dtype

    @staticmethod
    def _get_input_shape(input):
        if isinstance(input, torch.Tensor):
            shape = input.shape
        elif isinstance(input, dict):
            shape = input[list(input.keys())[0]].shape
        elif isinstance(input, (list, tuple, set)):
            shape = input[0].shape
        return shape

    @staticmethod
    def _get_dist_fns(distribution: List[str]) -> List[Callable]:
        """Returns the distribution functions.

        Arguments:
            distribution: List of distribution function names.

        Returns:
            Deserialized distribution functions.
        """
        return [
            getattr(torch, dist)
            if dist != 'linear' else partial(torch.mul, 1)
            for dist in distribution
        ]

    @staticmethod
    def _min_max_scaling(x: torch.Tensor, min: float, max: float) -> torch.Tensor:
        """Returns a min-max scaled tensor."""
        denominator = (torch.max(x) - torch.min(x))
        if torch.isclose(denominator, torch.zeros_like(denominator)):
            denominator = 1.0

        return (x - torch.min(x)) / denominator * (max - min) + min

    def forward(
            self,
            batch: Union[torch.Tensor, Sequence[torch.Tensor], Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Returns predefined reference points for the given input.

        Arguments:
            batch: Any sequence or dictionary of tensors.

        Returns:
            queries: Tensor of query reference points according to the
                instance arguments.
        """
        # Get input shape
        shape = self._get_input_shape(batch)

        # Get input data type
        dtype = self._get_input_dtype(batch)

        # Get input device
        device = self._get_input_device(batch)

        # Get initial unit queries per dimension [(N, ), ...] with len dim
        queries = [
            torch.linspace(0.0, 1.0, res, dtype=dtype, device=device) for res in self.resolution
        ]

        # Apply distribution function to unit queries
        queries = [dist_fn(query) for query, dist_fn in zip(queries, self._dist_fns)]

        # Scale queries to the desired value range
        queries = [
            self._min_max_scaling(query, mi, ma)
            for query, mi, ma in zip(queries, self.minimum, self.maximum)
        ]

        # Construct coordinate values from individual queries (N, dim)
        queries = torch.meshgrid(*tuple(queries), indexing='ij')
        queries = torch.stack([torch.flatten(q) for q in queries], dim=-1)

        # Match queries to the batch size (N, dim) -> (B, N, dim)
        queries = queries.repeat((shape[0], ) + (1, ) * queries.dim())

        # Transform queries
        queries = self.transformation(queries)

        return OrderedDict({'center': queries})


class DataAgnosticLinearQueries(DataAgnosticStaticQueries):
    def __init__(self,
                 resolution: List[int] = None,
                 minimum: List[float] = None,
                 maximum: List[float] = None,
                 transformation: nn.Module = None,
                 **kwargs):
        """Returns linear reference query points of arbitrary dimensionality.

        Arguments:
            resolution: Number of query points to generate for each dimension.
            minimum: Minimum value of the smalles reference point in each dimension.
            maximum: Maximum value of the largest reference point in each dimension.
        """
        super().__init__(resolution=resolution, minimum=minimum, maximum=maximum,
                         transformation=transformation, distribution='linear', **kwargs)

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        transformation = build_transformation(config['transformation'])
        return cls(
            config['resolution'],
            config['minimum'],
            config['maximum'],
            transformation=transformation
        )


def build_data_agnostic_query(name: str, *args, **kwargs):
    if 'static' in name.lower():
        return DataAgnosticStaticQueries.from_config(*args, **kwargs)
    if 'linear' in name.lower():
        return DataAgnosticLinearQueries.from_config(*args, **kwargs)
