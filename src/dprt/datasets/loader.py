from typing import Any, Dict, List, Tuple

import torch

from torch.utils.data import DataLoader, Dataset, default_collate

from dprt.utils.misc import as_list


def listed_collating(
        data: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """
    Attributes:
        data: List to data tuples consiting of input and target values.

    Returns:
        batch: Batched data consiting of a tuple of batched inputs and
            a list of targets.
    """
    # Split data into inputs and targets (list of tuples to tuple of lists)
    inputs, targets = zip(*data)

    # Ensure list data type
    inputs = as_list(inputs)
    targets = as_list(targets)

    # Convert tensors to batch of tensors
    inputs = default_collate(inputs)

    # Combine inputs and outputs
    batch = (inputs, targets)

    return batch


def load_listed(dataset: Dataset, config: Dict[str, Any]) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=config['train']['batch_size'],
        shuffle=config['train']['shuffle'],
        num_workers=config['computing']['workers'],
        collate_fn=listed_collating
    )
