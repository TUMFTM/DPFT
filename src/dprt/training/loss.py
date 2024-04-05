from __future__ import annotations  # noqa: F407

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import default_collate

from dprt.training.assigner import build_anassigner
from dprt.utils.bbox import get_box_corners
from dprt.utils.data import decollate_batch
from dprt.utils.iou import giou3d


def focal_loss(inputs: torch.Tensor, targets: torch.Tensor,
               alpha: float = 0.75, gamma: float = 2.0,
               reduction: str = "none") -> torch.Tensor:
    """Focal loss function.

    Reference: https://arxiv.org/abs/1708.02002

    Arguments:
        inputs: A float tensor of arbitrary shape (B, ...).
            The predictions for each example.
        targets: A float tensor with the same shape as inputs.
            Stores the binary classification label for each element
            in inputs (0 for the negative class and 1 for the positive class).
        alpha: Weighting factor in range (0, 1) to balance
            positive vs negative examples.
        gamma: Exponent of the modulating factor (1 - p_t) to
            balance easy vs hard examples.
        reduction: Reduction mode for the per class loss values.
            One of either none, sum or average.

    Retruns:
        loss: Loss tensor with the reduction option applied.
    """
    # Get cross entropy loss
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    # Get focal loss
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    # Balance loss
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Reduce loss (if required)
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class FocalLoss(nn.modules.loss._Loss):
    def __init__(self,
                 alpha: float = 0.75,
                 gamma: float = 2.0,
                 reduction: str = 'sum',
                 **kwargs):
        """Focal loss function.

        Reference: https://arxiv.org/abs/1708.02002

        Arguments:
            alpha: Weighting factor in range (0, 1) to balance
                positive vs negative examples.
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            reduction: Reduction mode for the per class loss values.
                One of either none, sum or average.
        """
        super().__init__(**kwargs)

        # Initialize instance attributes
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        # Check input arguments
        if self.reduction not in {'none', 'mean', 'sum'}:
            raise ValueError(
                    f"Invalid Value for arg 'reduction': '{self.reduction}"
                    f"\n Supported reduction modes: 'none', 'mean', 'sum'"
                )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Returns the focal loss value for the given input.

        Arguments:
            inputs: A float tensor of arbitrary shape (B, ...).
                The predictions for each example.
            targets: A float tensor with the same shape as inputs.
                Stores the binary classification label for each element
                in inputs (0 for the negative class and 1 for the positive class).

        Retruns:
            loss: Loss tensor with the reduction option applied.
        """
        return focal_loss(inputs, targets, self.alpha, self.gamma, self.reduction)


class GIoULoss(nn.modules.loss._Loss):
    def __init__(self,
                 reduction: str = 'sum',
                 **kwargs):
        """Generalized intersection over union loss.

        Reference: https://giou.stanford.edu/

        Arguments:
            reduction: Reduction mode for the per class loss values.
                One of either none, sum or average.
        """
        super().__init__(**kwargs)

        # Initialize instance attributes
        self.reduction = reduction

        # Check input arguments
        if self.reduction not in {'none', 'mean', 'sum'}:
            raise ValueError(
                    f"Invalid Value for arg 'reduction': '{self.reduction}"
                    f"\n Supported reduction modes: 'none', 'mean', 'sum'"
                )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Returns the generalized intersection over union loss.

        Arguments:
            inputs: A float tensor of shape (B, N, 8). The inputs
                represent bounding boxes with elements
                (x, y, z, l, w, h, sin a, cos a).
            targets: A float tensor with the same shape as inputs.

        Retruns:
            loss: Loss tensor with the reduction option applied.
        """
        # Get input shape
        B = inputs.shape[0]
        N = inputs.shape[1]

        # Get bounding box angles
        angle = torch.atan2(inputs[..., 6], inputs[..., 7])
        gt_angle = torch.atan2(targets[..., 6], targets[..., 7])

        # Get box corners
        corners = get_box_corners(inputs[..., :3], inputs[..., 3:6], angle)
        gt_corners = get_box_corners(targets[..., :3], targets[..., 3:6], gt_angle)

        # Get giou (giou is between -1 and 1) [0, 2]
        loss = 1 - torch.diagonal(giou3d(corners, gt_corners))

        # Reshape and scale to [0, 1]
        loss = loss.reshape(B, N) / 2

        # Reduce loss (if required)
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class SetCriterion(nn.Module):
    def __init__(self):
        """Set-to-Set loss

        Inspired by: Deformable-DETR

        losses: Dictionary of loss functions. Maps a
            loss name to a loss function.
        loss_inputs: Dictionary of loss input values.
                Maps a loss name to input (value) names.
        """
        super().__init__()

        self.losses = {
            "total_class": "total_focal_loss",
            "object_class": "object_focal_loss",
            "center": "l1_loss",
            "size": "l1_loss",
            "angle": "l1_loss"
        }

        self.loss_inputs = {
            "total_class": ["class"],
            "object_class": ["class"],
            "center": ["center"],
            "size": ["size"],
            "angle": ["angle"]
        }

    @staticmethod
    def _batched_index_select(batch: torch.Tensor, dim: int, inds: torch.Tensor) -> torch.Tensor:
        """Returns elements of a batched tensor given their indices.

        Arguments:
            batch: The batched input tensor of shape (B, N, M)
            dim: The dimension in which we index.
            inds: The 2D tensor containing the indices to
                index with shape (B, N)
        """
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), batch.size(2))
        out = batch.gather(dim, inds)
        return out

    @staticmethod
    def _dstack_dict(dictionary: Dict[str, torch.Tensor], keys: List[str]) -> torch.Tensor:
        """Retruns a stacked tensor from a dict of tensors.

        Arguments:
            dictionary: Dictionary of tensors with shape (B, N, C1),
                where C1 can vary across the tensors.
            keys: Dictionary keys of the tensors to stack.

        Returns:
            Stacked tensor with shape (B, N, C2), where
                C2 = sum(C1).
        """
        return torch.dstack([dictionary[k] for k in keys])

    def object_focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor,
                          indices: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Returns an aggregated focal loss value for all associated predictions.

        Calculates a focal loss value between the associated predictions
        and ground truth.

        Arguments:
            inputs: Tensor of model predictions with shape (B, N, C1)
            targets: Tensor of ground truth values with shape (B, M, C1)
            indices: Tuple of indices for the association between prediction
                and ground truth values with shape ((B, M), (B, M))

        Returns:
            loss: Loss value with shape (B, )
        """
        # Split indices into prediction and target indices
        i, j = indices

        # Get number of queries
        N = inputs.shape[1]

        # Get number of ground truth objects
        M = j.numel()

        # Calculate L1 loss between the matched inputs and targets
        loss = focal_loss(
            self._batched_index_select(inputs, dim=1, inds=i),
            self._batched_index_select(targets, dim=1, inds=j),
            reduction='none'
        )

        # Average loss
        loss = (loss.mean(1).sum() / M) * N

        return loss

    def total_focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor,
                         indices: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Returns an aggregated focal loss value for all predictions.

        Calculates a focal loss value for all predictions by matching all
        unassociated predictions to the background class.

        Note: It is assumed that the first ("zerost") class is the None class!

        Arguments:
            inputs: Tensor of model predictions with shape (B, N, C1)
            targets: Tensor of ground truth values with shape (B, M, C1)
            indices: Tuple of indices for the association between prediction
                and ground truth values with shape ((B, M), (B, M))

        Returns:
            loss: Loss value with shape (B, )
        """
        # Get input shape and device
        B, N, C = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        # Split indices into prediction and target indices
        i, j = indices

        # Get number of ground truth objects
        M = j.numel()

        # Initialize target class probabilities (one hot encoded)
        target_one_hot: torch.Tensor = F.one_hot(
            torch.zeros((B, N), dtype=torch.int64, device=device),
            num_classes=C
        )
        target_one_hot = target_one_hot.type(dtype)

        # Assign ground truth labels to the indices of the assigned predictions
        index = i.unsqueeze(2).expand(i.size(0), i.size(1), C)
        target_one_hot.scatter_(dim=1, index=index, src=targets)

        # Get focal loss value
        loss = focal_loss(inputs, target_one_hot, reduction='none')

        # Average loss
        loss = (loss.mean(1).sum() / M) * N

        return loss

    def l1_loss(self, inputs: torch.Tensor, targets: torch.Tensor,
                indices: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Returns the mean L1 loss for the given inputs.

        Arguments:
            inputs: Tensor of model predictions with shape (B, N, C1)
            targets: Tensor of ground truth values with shape (B, M, C1)
            indices: Tuple of indices for the association between prediction
                and ground truth values with shape ((B, M), (B, M))

        Returns:
            loss: Loss value with shape (B, )
        """
        # Split indices into prediction and target indices
        i, j = indices

        # Calculate L1 loss between the matched inputs and targets
        loss = F.l1_loss(
            self._batched_index_select(inputs, dim=1, inds=i),
            self._batched_index_select(targets, dim=1, inds=j),
            reduction='mean'
        )

        return loss

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        indices: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Returns the set-to-set loss values.

        Arguments:
            inputs: Dictionary of model predictions. Each value of the
                dictionary is a tensor with shape (B, N, C).
            targets: Dictionary of ground truth values. Each value of the
                dictionary is a tensor with shape (B, M, C).
            indices: Tuple of indices for the association between prediction
                and ground truth values with shape ((B, M), (B, M))

        Returns:
            losses: Dictionary of loss values with shape (B, ).
        """
        # Get losses
        losses = {
            name: getattr(self, func)(
                self._dstack_dict(inputs, self.loss_inputs[name]),
                self._dstack_dict(targets, [f"gt_{n}" for n in self.loss_inputs[name]]),
                indices
            )
            for name, func in self.losses.items()
        }

        return losses


class Loss(nn.modules.loss._Loss):
    def __init__(self,
                 anassigner: nn.Module = None,
                 criterion: nn.Module = None,
                 losses: Dict[str, nn.modules.loss._Loss] = None,
                 loss_inputs: Dict[str, List[str]] = None,
                 loss_weights: Dict[str, float] = None,
                 reduction: str = 'mean',
                 **kwargs):
        """Loss module.

        Arguments:
            anassigner: Anassigner to match model predictions
                with ground truth values.
            losses: Dictionary of loss functions. Maps a
                loss name to a loss function.
            loss_inputs: Dictionary of loss input values.
                Maps a loss name to input (value) names.
            loss_weights: Dictionary of loss weights. Maps a
                loss name to a loss weight.
            reduction: Reduction mode for the per batch loss values.
                One of either none, sum or mean.
        """
        # Initialize base class
        super().__init__(**kwargs)

        # Check input arguments
        if anassigner is not None:
            assert criterion is not None

        if reduction not in {'none', 'mean', 'sum'}:
            raise ValueError(
                    f"Invalid Value for arg 'reduction': '{self.reduction}"
                    f"\n Supported reduction modes: 'none', 'mean', 'sum'"
                )

        # Initialize loss instance
        self.losses = losses if losses is not None else {}
        self.loss_inputs = loss_inputs if loss_inputs is not None else {}
        self.loss_weights = loss_weights if loss_weights is not None else {}
        self.anassigner = anassigner
        self.criterion = criterion
        self.reduction = reduction

        # Get reduction function
        if self.reduction != 'none':
            self.reduction_fn = getattr(torch, self.reduction)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Loss:  # noqa: F821
        anassigner = None
        criterion = None
        losses = None
        loss_inputs = config.get('loss_inputs')
        loss_weights = config.get('loss_weights')
        reduction = config.get('reduction', 'mean')

        if 'anassigner' in config:
            anassigner = build_anassigner(config['anassigner'], config)

        if 'criterion' in config:
            criterion = _get_loss(config['criterion'])

        if 'losses' in config:
            losses = {k: _get_loss(v) for k, v in config['losses'].items()}

        return cls(
            anassigner=anassigner,
            criterion=criterion,
            losses=losses,
            loss_inputs=loss_inputs,
            loss_weights=loss_weights,
            reduction=reduction
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
    def _dstack_dict(dictionary: Dict[str, torch.Tensor], keys: List[str]) -> torch.Tensor:
        """Retruns a stacked tensor from a dict of tensors.

        Arguments:
            dictionary: Dictionary of tensors with shape (B, N, C1),
                where C1 can vary across the tensors.
            keys: Dictionary keys of the tensors to stack.

        Returns:
            Stacked tensor with shape (B, N, C2), where
                C2 = sum(C1).
        """
        return torch.dstack([dictionary[k] for k in keys])

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns the loss given a prediction and ground truth.

        Arguments:
            inputs: Dictionary of model predictions with shape (B, N, C).
            targets: Dictionary of ground truth values with shape (B, M, C).

        Returns:
            total_loss: Sum over all loss values with shape (B, ).
            losses: Dictionary of loss values with shape (B, ).
        """
        # Get input data device and dtype
        device = self._get_input_device(inputs)
        dtype = self._get_input_dtype(inputs)

        # Initialize losses
        batch_losses = []

        # Decollate inputs
        inputs: List[Dict[str, torch.Tensor]] = decollate_batch(inputs, detach=False, pad=False)

        # Get loss for each item in the batch
        for input, target in zip(inputs, targets):
            # Append zero and continue if no targets are present
            if not all([t.numel() for t in target.values()]):
                batch_losses.append({
                    name: torch.zeros((), device=device, dtype=dtype, requires_grad=True)
                    for name in self.loss_weights.keys()
                })
                continue

            # Insert dummy batch dimension
            input = {k: v.unsqueeze(0) for k, v in input.items()}
            target = {k: v.unsqueeze(0) for k, v in target.items()}

            if self.anassigner is not None:
                # Get assignment
                i, j = self.anassigner(input, target)

                # Apply loss criterion
                losses = self.criterion(input, target, indices=(i, j))

            else:
                # Get loss values
                losses = {
                    name: func(
                        self._dstack_dict(input, self.loss_inputs[name]),
                        self._dstack_dict(target, [f"gt_{n}" for n in self.loss_inputs[name]])
                    )
                    for name, func in self.losses.items()
                }

            # Weight loss values
            for k, weight in self.loss_weights.items():
                losses[k] *= weight

            # Add losses to the batch
            batch_losses.append(losses)

        # Catch no loss configuration
        if not self.losses:
            return (torch.zeros(1, device=device, dtype=dtype, requires_grad=True),
                    {k: torch.zeros(1, device=device, dtype=dtype) for k in self.losses.keys()})

        # Collate losses (revert decollating)
        batch_losses: Dict[str, torch.Tensor] = default_collate(batch_losses)

        # Reduce batch losses
        if self.reduction != 'none':
            batch_losses = {k: self.reduction_fn(v) for k, v in batch_losses.items()}

        # Get total loss
        total_loss = torch.stack(tuple(batch_losses.values())).sum(dim=-1)

        return total_loss, batch_losses


def _get_loss(name: str) -> nn.modules.loss._Loss:
    """Returns a pytorch or custom loss function given its name.

    Attributes:
        name: Name of the loss function (class).

    Returns:
        Instance of a loss function.
    """
    try:
        return getattr(nn, name)()
    except AttributeError:
        return globals()[name]()
    except Exception as e:
        raise e


def build_loss(*args, **kwargs):
    return Loss.from_config(*args, **kwargs)
