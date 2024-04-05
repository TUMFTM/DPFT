from typing import Optional, Union, Tuple

import torch

from torch import nn


class Unary1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dilation: Optional[Union[int, Tuple[int, ...]]] = 1,
                 groups: Optional[int] = 1,
                 bias: Optional[bool] = True,
                 channels_last: Optional[bool] = True,
                 **kwargs):
        """Unary 1D layer.

        Arguments:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections from input channels to
                output channels.
            bias: If True, adds a learnable bias to the output.
            channels_last: Whether the input data is in channel last
                or channel first format.

        """
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.channels_last = channels_last

        self.conv1d = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1,
                                stride=1, padding=0, dilation=self.dilation,
                                groups=self.groups, bias=self.bias)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Arguments:
            batch: Batch of size (B, C, N) if channel_last is False
                or (B, N, C) if channel_last is True.
        """
        if self.channels_last:
            batch = batch.permute(0, 2, 1)

        batch = self.conv1d(batch)

        if self.channels_last:
            batch = batch.permute(0, 2, 1)

        return batch


class Unary2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dilation: Optional[Union[int, Tuple[int, ...]]] = 1,
                 groups: Optional[int] = 1,
                 bias: Optional[bool] = True,
                 channels_last: Optional[bool] = False,
                 **kwargs):
        """Unary 2D layer.

        Arguments:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections from input channels to
                output channels.
            bias: If True, adds a learnable bias to the output.
            channels_last: Whether the input data is in channel last
                or channel first format.

        """
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.channels_last = channels_last

        self.conv2d = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1,
                                stride=1, padding=0, dilation=self.dilation,
                                groups=self.groups, bias=self.bias)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Arguments:
            batch: Batch of size (B, C, H, W) if channel_last is False
                or (B, W, H, C) if channel_last is True.
        """
        if self.channels_last:
            batch = batch.permute(0, 3, 2, 1)

        batch = self.conv2d(batch)

        if self.channels_last:
            batch = batch.permute(0, 3, 2, 1)

        return batch
