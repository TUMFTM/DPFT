from __future__ import annotations  # noqa: F407

from collections import OrderedDict
from typing import Any, Dict, List, Union, Sequence

import torch

from torch import nn

from dprt.models.utils.transformations import build_transformation


class LearnableQueries(nn.Module):
    def __init__(self,
                 resolution: List[int] = None,
                 minimum: List[float] = None,
                 maximum: List[float] = None,
                 q_init: str = None,
                 transformation: nn.Module = None,
                 **kwargs):
        """Returns learnable reference query points of arbitrary dimensionality.

        Arguments:
            resolution: Number of query points to generate for each dimension.
            minimum: Minimum value of the smalles reference point in each dimension.
            maximum: Maximum value of the largest reference point in each dimension.
            q_init: Query point initialization method.
            transformation: Transformation (coordinate transformation) to apply
                to the query points.
        """
        # Initialize parent class
        super().__init__()

        # Initialize instance attributes
        self.resolution = resolution if resolution is not None else []
        self.minimum = minimum if minimum is not None else []
        self.maximum = maximum if maximum is not None else []
        self.q_init = q_init if q_init is not None else 'uniform_'

        if transformation is not None:
            self.transformation = transformation
        else:
            self.transformation = nn.Identity()

        # Check input arguments
        assert len(self.resolution) == len(self.minimum) == len(self.maximum)

        # Initialize queries (N, dim)
        queries = torch.empty((torch.prod(torch.tensor(self.resolution)), len(self.resolution)))
        self.queries = nn.Parameter(queries)

        self.reset_parameters()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> LearnableQueries:  # noqa: F821
        transformation = build_transformation(config.get('transformation'))
        return cls(
            config['resolution'],
            config['minimum'],
            config['maximum'],
            config.get('q_init'),
            transformation=transformation,
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

    def reset_parameters(self) -> None:
        """Initialize query values."""
        if self.q_init == 'uniform_':
            for i, (mi, ma) in enumerate(zip(self.minimum, self.maximum)):
                torch.nn.init.uniform_(self.queries[..., i], a=mi, b=ma)
        else:
            getattr(torch.nn.init, self.q_init)(self.queries)

    def forward(
            self,
            batch: Union[torch.Tensor, Sequence[torch.Tensor], Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Returns predefined reference points for the given input.

        Arguments:
            batch: Any sequence or dictionary of tensors.

        Returns:
            queries: Tensor of learnable query reference points according
                to the instance arguments.
        """
        # Get input shape
        shape = self._get_input_shape(batch)

        # Get batch size
        B = shape[0]

        # Adjust query dimensions (N, dim) -> (B, N, dim)
        queries = self.queries.unsqueeze(0).repeat(B, 1, 1)

        # Transform queries
        queries = self.transformation(queries)

        return OrderedDict({'center': queries})


def build_learnable_query(name: str, *args, **kwargs):
    return LearnableQueries.from_config(*args, **kwargs)
