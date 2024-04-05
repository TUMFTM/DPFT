from __future__ import annotations  # noqa: F407

import math

from collections import OrderedDict
from typing import Any, Dict

import torch
from torch import nn


class SinusoidalEmbedding(nn.Module):
    """Position encoding with sine and cosine functions.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
    """
    def __init__(self,
                 num_feats: int,
                 temperature: int = 10000,
                 normalize: bool = False,
                 scale: float = 2 * math.pi,
                 eps: float = 1e-6,
                 offset: float = 0.,
                 **kwargs):
        # Initialize base class
        super().__init__()

        # Check normalization setting
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'

        # Initialize instance arguments
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> SinusoidalEmbedding:  # noqa: F821
        return cls(**config)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Applies a sinusoidal positional embedding to the imput batch.

        Args:
            batch: Batched input tensor with shape (B, H, W, C).

        Returns:
            batch: Embedded input tensor with shape (B, H, W, C).
        """
        # Get input shape, device and dtype
        B, H, W, _ = batch.shape
        device = batch.device
        dtype = batch.dtype

        # Get mask
        mask = torch.zeros((B, H, W), device=device, dtype=torch.int)

        # Invert mask (logical_not)
        not_mask = 1 - mask

        y_embed = not_mask.cumsum(1, dtype=dtype)
        x_embed = not_mask.cumsum(2, dtype=dtype)

        if self.normalize:
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = torch.arange(
            self.num_feats, dtype=dtype, device=device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)

        # Positional embedding with shape (B, H, W, num_feats)
        batch += pos_x
        batch += pos_y

        return batch


class MultiLevelSinusoidalEmbedding(nn.Module):
    def __init__(self,
                 n_levels: int = 1,
                 **kwargs):
        """
        Arguments:
            n_levels: Number of feature maps in the
                input batches.
        """
        super().__init__()

        self.n_levels = n_levels

        # Initialize fusion layers
        self.embedding_layers = nn.ModuleDict({
            'embedding' + str(i):
            SinusoidalEmbedding(**kwargs)
            for i in range(self.n_levels)
        })

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> MultiLevelSinusoidalEmbedding:  # noqa: F821
        return cls(**config)

    def forward(self, batches: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
        """
        Arguments:
            batches: Ordered dictionary that contains n_levels feature maps
                with shape (B, H, W, C), whereas H, W and C can be
                different for every feature map.

        Returns:
            outputs: Ordered dictionary that contains n_levels embedded
                feature maps with the same shapes as in batches.
        """
        iterator = zip(batches.items(), self.embedding_layers.values())

        outputs = OrderedDict(
            {k: layer(batch) for (k, batch), layer in iterator}
        )
        return outputs


def build_sinusoidal_embedding(*args, **kwargs) -> nn.Module:
    return MultiLevelSinusoidalEmbedding.from_config(*args, **kwargs)
