# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
from __future__ import annotations  # noqa: F407

from typing import Any, Dict, Tuple

import torch
import numpy as np

from scipy.optimize import linear_sum_assignment
from torch import nn

from dprt.utils.bbox import get_box_corners
from dprt.utils.iou import giou3d


class HungarianAnassigner(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this,
    in general, there are more predictions than targets. In this case,
    we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 loss_weights: Dict[str, float] = None,
                 giou_weight: float = 1.0,
                 **kwargs):
        """Creates the matcher

        Arguments:
            loss_weights: Dictionary of loss weights. Mapping a
                model prediction name (in inputs) to a loss
                weight.
        """
        super().__init__()

        self.loss_weights = loss_weights
        self.giou_weight = giou_weight

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> HungarianAnassigner:  # noqa: F821
        loss_weights = config.get('loss_weights')
        return cls(
            loss_weights=loss_weights
        )

    def forward(self,
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Performs the matching

        The shape variable correspond to:
            B: Batch size
            N: Number of predicted objects
            M: Number of target objects
            C: Number of classes

        Arguments:
            outputs: This is a dict that contains at least these entries:
                "class": Bounding box class probabilities of shape (B, N, C)
                "center": Bounding box center coordinates of shape (B, N, C1).
                "size": Bounding box size values of shape (B, N, C2).
                "angle": Bounding box orientation values of shape (B, N, C3).

            targets: This is a dict of targets that contains at least these entries:
                "gt_class": Bounding box class probabilities of shape (B, M, C)
                "gt_center": Bounding box center coordinates of shape (B, M, C1).
                "gt_size": Bounding box size values of shape (B, M, C2).
                "gt_angle": Bounding box orientation values of shape (B, M, C3).

        Returns:
            A tuple of tensors (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                    with shape (B, M)
                - index_j is the indices of the corresponding selected targets (in order)
                    with shape (B, M)
        """
        with torch.no_grad():
            # Get output shape
            bs, num_queries = outputs["class"].shape[:2]

            # Get number of bounding boxes per batch
            sizes = [targets['gt_class'].shape[1]]

            # Get output device
            device = outputs["class"].device

            # We flatten to compute the cost matrices in a batch
            out_class = outputs["class"].flatten(0, 1)
            out_center = outputs["center"].flatten(0, 1)
            out_size = outputs["size"].flatten(0, 1)
            out_angle = outputs["angle"].flatten(0, 1)

            gt_class = targets["gt_class"].flatten(0, 1)
            gt_center = targets["gt_center"].flatten(0, 1)
            gt_size = targets["gt_size"].flatten(0, 1)
            gt_angle = targets["gt_angle"].flatten(0, 1)

            gt_ids = torch.argmax(gt_class, dim=-1)

            # Compute the classification cost
            cost_class = -out_class[:, gt_ids]

            # Compute the L1 cost between boxes
            cost_center = torch.cdist(out_center, gt_center, p=1)
            cost_size = torch.cdist(out_size, gt_size, p=1)
            cost_angle = torch.cdist(out_angle, gt_angle, p=1)

            # Compute the giou cost betwen boxes
            out_angle = torch.atan2(outputs["angle"][..., 0], outputs["angle"][..., 1])
            gt_angle = torch.atan2(targets["gt_angle"][..., 0], targets["gt_angle"][..., 1])
            out_corners = get_box_corners(outputs["center"], outputs["size"], out_angle)
            gt_corners = get_box_corners(targets["gt_center"], targets["gt_size"], gt_angle)
            cost_giou = -giou3d(out_corners, gt_corners)

            # Final cost matrix
            C = self.loss_weights['total_class'] * cost_class \
                + self.loss_weights['center'] * cost_center \
                + self.loss_weights['size'] * cost_size \
                + self.loss_weights['angle'] * cost_angle \
                + self.giou_weight * cost_giou

            # Reconstruct original shape
            C = C.view(bs, num_queries, -1).cpu()

            # Match predictions and ground truth
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            i, j = zip(*indices)

            index_i = torch.from_numpy(np.stack(i)).to(device=device)
            index_j = torch.from_numpy(np.stack(j)).to(device=device)
            return index_i, index_j


def build_anassigner(name: str, *args, **kwargs):
    if 'hungarian' in name.lower():
        return HungarianAnassigner.from_config(*args, **kwargs)
