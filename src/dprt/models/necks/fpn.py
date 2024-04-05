from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

import torch
import torchvision

from torch import nn
from torchvision.ops import FeaturePyramidNetwork


class FPN(nn.Module):
    def __init__(self,
                 in_channels_list: List[int],
                 out_channels: int,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 channel_last: bool = True,
                 **kwargs):
        """Feature Pyramid Network

        Arguments:
            in_channels_list: Number of channels for each feature map
                that is passed to the module.
            out_channels: Number of channels of the FPN representation.
            norm_layer: Module specifying the normalization layer to use.
            channel_last: Channle format of the given input data.
                True if the input is given in channel last format,
                False otherwise.
        """
        super().__init__(**kwargs)

        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.channel_last = channel_last
        self.norm_layer = norm_layer

        if self.norm_layer is not None:
            self.norm_layer = self._get_norm_layer(norm_layer)

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.in_channels_list,
            out_channels=self.out_channels,
            norm_layer=self.norm_layer
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(
            config['in_channels_list'],
            config['out_channels'],
            config.get('norm_layer'),
        )

    @staticmethod
    def _get_norm_layer(name: str) -> nn.Module:
        try:
            return getattr(torchvision.ops, name)
        except AttributeError:
            return getattr(torch.nn, name)
        except Exception as e:
            raise e

    @staticmethod
    def _to_channel_first(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return OrderedDict({k: v.movedim(-1, 1) for k, v in batch.items()})

    @staticmethod
    def _to_channel_last(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return OrderedDict({k: v.movedim(1, -1) for k, v in batch.items()})

    def forward(self,
                batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Adjust channel format
        if self.channel_last:
            batch = self._to_channel_first(batch)

        # Align features
        batch = self.fpn(batch)

        # Adjust channel format
        if self.channel_last:
            batch = self._to_channel_last(batch)

        return batch


def build_fpn(name: str, *args, **kwargs):
    if 'fpn' in name.lower():
        return FPN.from_config(*args, **kwargs)
