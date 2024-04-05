from __future__ import annotations  # noqa: F407

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Tuple

import torch
from torch import nn

from dprt.models.backbones import build_backbone
from dprt.models.necks import build_neck
from dprt.models.embeddings import build_embedding
from dprt.models.queries import build_querent
from dprt.models.fusers import build_fuser
from dprt.models.heads import build_head


def _build_module(build_fn: Callable, module_name: str,
                  config: Dict[str, Any], computing: Dict[str, Any],
                  *args, **kwargs) -> nn.Module:
    """Retruns an module instance given its configuration.

    Arguments:
        build_fn: Build function for the particular module.
        module_name: Name of the module.
        config: Configuration of the module instance.

    Returns:
        Module instance or Identity module if module_name is not in config.
    """
    # Get module configuration
    module: Dict[str, Any] = config.get(module_name)

    if module is not None:
        return build_fn(
            module['name'], dict(computing | module), *args, **kwargs
        )

    return None


def _build_modules(build_fn: Callable, module_name: str,
                   config: Dict[str, Any], computing: Dict[str, Any],
                   *args, **kwargs) -> Dict[str, nn.Module]:
    """Retruns a dict of module instances given their configuration.

    Arguments:
        build_fn: Build function for the modules.
        module_name: Name of the module (parent module).
        config: Configuration of the module instances.

    Returns:
        Dict of module instances or None if module_name is not in config.
    """
    # Get modules configuration
    modules: Dict[str, Any] = config.get(module_name)

    # Build modules
    if modules is not None:
        return {
            k: _build_module(build_fn, k, modules, computing, *args, **kwargs)
            for k in modules.keys()
        }

    return None


class DPRT(nn.Module):
    def __init__(self,
                 inputs: List[str],
                 skiplinks: Dict[str, bool] = None,
                 backbones: Dict[str, nn.Module] = None,
                 necks: Dict[str, nn.Module] = None,
                 embeddings: Dict[str, nn.Module] = None,
                 querent: nn.Module = None,
                 fuser: nn.Module = None,
                 head: nn.Module = None,
                 **kwargs):
        """Dual Perspective Radar Transformer

        Arguments:
            inputs: List of input data names representing multiple
                perspectives (modalities).
            skiplinks: Dict of boolean values specifying whether to pass
                the raw data of the corresponding input to the fusion module.
            backbones: Dict of backbone modules used for the feature extraction
                of the corresponding input data.
            necks: Dict of neck modules used for the feature alignment of
                the extracked features and input data (if skiplink).
            embeddings: Dict of embedding modules used for positional encoding
                of the feature maps of the corresponding inputs.
            querent: Query reference point generation module.
            fuser: Fusion module used to fuse the different views (modalities).
            head: Head model used to generate the model prediction.
        """
        # Initialize base class
        super().__init__()

        # Initialize instance attributes
        self.inputs = inputs
        self.skiplinks = skiplinks if skiplinks is not None else {}
        self.backbones = backbones if backbones is not None else {}
        self.necks = necks if necks is not None else {}
        self.embeddings = embeddings if embeddings is not None else {}

        # Initialize unspecified submodules
        self.skiplinks = {input: self.skiplinks.get(input, False) for input in inputs}
        self.backbones = self._init_unspecified(self.backbones)
        self.necks = self._init_unspecified(self.necks)
        self.embeddings = self._init_unspecified(self.embeddings)
        self.querent = self._module_or_identity(querent)
        self.fuser = self._module_or_identity(fuser)
        self.head = self._module_or_identity(head)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> DPRT:  # noqa: F821
        # Get general subconfigs
        computing: Dict[str, Any] = config['computing']
        model: Dict[str, Any] = config['model']

        # Build fuser-head combination
        head = _build_module(build_head, 'head', model, computing)
        fuser = _build_module(build_fuser, 'fuser', model, computing, head=head)

        return cls(
            inputs=model.get('inputs'),
            skiplinks=model.get('skiplinks'),
            backbones=_build_modules(build_backbone, 'backbones', model, computing),
            necks=_build_modules(build_neck, 'necks', model, computing),
            embeddings=_build_modules(build_embedding, 'embeddings', model, computing),
            querent=_build_module(build_querent, 'querent', model, computing),
            fuser=fuser,
            head=head
        )

    def _init_unspecified(self, submodule: Dict[str, nn.Module]) -> Dict[str, nn.Module]:
        """Returns a dict of module instances

        Arguments:
            submodule: Dictionary of submodules.

        Returns:
            Dictionary of modules with one module for each input.
            If no module is provided the Identity module is used.
        """
        return nn.ModuleDict(
            {input: self._module_or_identity(submodule.get(input)) for input in self.inputs}
        )

    @staticmethod
    def _module_or_identity(module: nn.Module = None) -> nn.Module:
        """Returns the given module or the Identity module.

        Arguments:
            module: Module to return.

        Returns:
            module: The given module or the Identity module,
                if the given module is None.
        """
        if module is not None:
            return module
        return nn.Identity()

    @staticmethod
    def _add_raw_data(features: OrderedDict[str, torch.Tensor],
                      raw_data: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        """Returns the feature dict extend by the raw data.

        Arguments:
            features: Dictionary of feature tensors.
            raw_data: Raw data tensor to insert into the feature dict.

        Returns:
            features: Dictionary of feature tensors with the raw data
                inserted in the front.
        """
        features['0'] = raw_data
        features.move_to_end('0', last=False)
        return features

    @staticmethod
    def _get_projetions(inputs: List[str],
                        batch: Dict[str, torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Returns a tuple of projection matrices for each input.

        Arguments:
            inputs: List of input names.
            batch: Batch of input data containing the
                transformation and projetion matirces.

        Returns:
            A list of tuples containing a transformation
            and projection matirx for every input.
        """
        return [
            (batch[f'label_to_{input}_t'], batch[f'label_to_{input}_p'])
            for input in inputs
        ]

    def forward(self, batch: Dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
        """Returns the DPRT prediction based on the given input.

        Arguments:
            batch: Dictionary of batched model input tensors. For every input the
                dict must at least contain:
                "input_name": Input data with shape (B, H, W, C).
                "label_to_input_name_t": A transformation matrix with shape (B, 4, 4).
                "label_to_input_name_p": A projection matrix with shape (B, 4, 4).
                "input_name_shape": Original shape of the input data with shape (H, W, C).

        Returns:
            out: Dictionary of batched model predictions. The content of the dict is
                defined by the head module.
        """
        # Get input data shapes in channel last format (B, H, W, C)
        shapes = {input: batch[f"{input}_shape"] for input in self.inputs}

        # Feature extraction
        features = {input: self.backbones[input](batch[input]) for input in self.inputs}

        # Add input features (skip link)
        features = {
            input: self._add_raw_data(features[input], batch[input])
            for input in self.inputs if self.skiplinks[input]
        }

        # Feature alignment
        features = {input: self.necks[input](features[input]) for input in self.inputs}

        # Positional embedding
        features = {input: self.embeddings[input](features[input]) for input in self.inputs}

        # Get (global) reference points to query
        out = self.querent(batch)

        # Sensor fusion
        out = self.fuser(
            batch=[features[input] for input in self.inputs],
            shape=[shapes[input][:, :2] for input in self.inputs],
            projection=self._get_projetions(self.inputs, batch),
            out=out
        )

        return out


def build_dprt(*args, **kwargs):
    return DPRT.from_config(*args, **kwargs)
