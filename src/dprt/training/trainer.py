from __future__ import annotations  # noqa: F407

import datetime
import os
import os.path as osp

from typing import Any, Dict, Iterable, List

import torch

from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

from dprt.evaluation.metric import build_metric
from dprt.training.optimizer import build_optimizer
from dprt.training.loss import build_loss
from dprt.training.scheduler import build_scheduler


class CentralizedTrainer():
    def __init__(self,
                 epochs: int = 1,
                 optimizer: torch.optim.Optimizer = None,
                 loss: torch.nn.modules.loss._Loss = None,
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 metric: torch.nn.modules.loss._Loss = None,
                 device: str = None,
                 logging: str = None,
                 evaluating: int = 1):
        """
        Arguments:
            logging: Logging frequency. One of either None,
                step or epoch.
            evaluating: Evaluation frequency. A value of
                -1 means no evaluation, 0 means an evaluation
                after every step and evey value > 0 descibes the
                number of epoch after which an evaluation is executed.
        """
        # Initialize instance arrtibutes
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss_fn = loss
        self.scheduler = scheduler
        self.eval_fn = metric
        self.device = device
        self.logging = logging
        self.evaluating = evaluating

    @classmethod
    def from_config(cls,
                    config: Dict[str, Any],
                    *args,
                    **kwargs) -> CentralizedTrainer:  # noqa: F821
        # Get trainer atributes
        epochs = config['train']['epochs']
        optimizer = build_optimizer(
            config['train']['optimizer'].pop('name'),
            **config['train']['optimizer']
        )
        loss = build_loss(
            config['train']
        )
        scheduler = build_scheduler(
            config['train']['scheduler'].pop('name'),
            **config['train']['scheduler']
        )
        metric = build_metric(
            config['evaluate']
        )
        device = torch.device(config['computing']['device'])
        logging = config['train'].get('logging')

        return cls(
            epochs=epochs,
            optimizer=optimizer,
            loss=loss,
            scheduler=scheduler,
            metric=metric,
            device=device,
            logging=logging
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.train(*args, **kwargs)

    @staticmethod
    def log_scalars(writer, scalars: Dict[str, Any], epoch: int, prefix: str = None) -> None:
        # Get prefix
        prefix = f"{prefix}/" if prefix is not None else ""

        # Add scalar values
        for name, scalar in scalars.items():
            writer.add_scalar(prefix + name, scalar, epoch)

    @staticmethod
    def _dict_to(data: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
        return {k: v.to(device) for k, v in data.items()}

    def train_one_epoch(self, epoch: int, model: torch.nn.Module,
                        data_loader: Iterable, optimizer: torch.optim.Optimizer,
                        writer: SummaryWriter = None) -> None:
        # Make sure gradient tracking is on
        model.train()

        # Initialize epoch logs
        scalars = {}

        for i, (data, labels) in enumerate(data_loader):
            # Log learning rate
            if self.logging == 'step':
                writer.add_scalar('train/learning_rate',
                                  optimizer.param_groups[0]['lr'],
                                  i + epoch * len(data_loader))

            # Load data and labels (to device)
            labels: List[Dict[str, torch.Tensor]] = \
                [self._dict_to(label, self.device) for label in labels]
            data: Dict[str, torch.Tensor] = \
                self._dict_to(data, self.device)

            # Zero gradients
            optimizer.zero_grad()

            # Make prediction
            output = model(data)

            # Compute the loss and its gradients
            loss, losses = self.loss_fn(output, labels)

            # Adjust weights
            if loss > 0:
                loss.backward()
                optimizer.step()

            # Evaluate model output
            metrics = self.eval_fn(output, labels)

            # Add prefix to loss values (logging)
            losses = {f'loss_{k}': v for k, v in losses.items()}
            losses['loss'] = loss

            # Log training step
            if self.logging == 'step':
                self.log_scalars(writer, losses, i + epoch * len(data_loader), 'train')
                self.log_scalars(writer, metrics, i + epoch * len(data_loader), 'train')

            # Add values to epoch log
            if self.logging == 'epoch':
                for k, v in losses.items():
                    scalars[k] = scalars.get(k, 0) + v
                for k, v in metrics.items():
                    scalars[k] = scalars.get(k, 0) + v

        if self.logging == 'epoch':
            # Average epoch logs
            scalars = {k: v / (i + 1) for k, v in scalars.items()}

            # Write epoch logs
            self.log_scalars(writer, scalars, epoch, 'train')
            writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)

    @torch.no_grad()
    def validate_one_epoch(self, epoch: int, model: torch.nn.Module, data_loader: Iterable,
                           writer: SummaryWriter = None) -> Dict[str, float]:
        # Make sure the model is in evaluation mode
        model.eval()
        self.loss_fn.eval()

        # Initialize epoch logs
        scalars = {}

        for i, (data, labels) in enumerate(data_loader):
            # Load data and labels (to device)
            labels: List[Dict[str, torch.Tensor]] = \
                [self._dict_to(label, self.device) for label in labels]
            data: Dict[str, torch.Tensor] = \
                self._dict_to(data, self.device)

            # Make prediction
            output = model(data)

            # Compute the loss and its gradients
            loss, losses = self.loss_fn(output, labels)

            # Evaluate model output
            metrics = self.eval_fn(output, labels)

            # Add prefix to loss values (logging)
            losses = {f'loss_{k}': v for k, v in losses.items()}
            losses['loss'] = loss

            # Log training step
            if self.logging == 'step':
                self.log_scalars(writer, losses, i + epoch * len(data_loader), 'val')
                self.log_scalars(writer, metrics, i + epoch * len(data_loader), 'val')

            # Add values to epoch log
            for k, v in losses.items():
                scalars[k] = scalars.get(k, 0) + v
            for k, v in metrics.items():
                scalars[k] = scalars.get(k, 0) + v

        if self.logging == 'epoch':
            # Average epoch logs
            scalars = {k: v / (i + 1) for k, v in scalars.items()}

            # Write epoch logs
            self.log_scalars(writer, scalars, epoch, 'val')

        return {'loss': scalars['loss']}

    def train(self, model: torch.nn.Module, data_loader: Iterable, val_loader: Iterable = None,
              start_epoch: int = 0, timestamp: str = None, dst: str = None) -> None:
        # Load model (to device)
        model.to(self.device)

        # Get current timestamp
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]

        # Create checkpoint directory
        os.makedirs(osp.join(dst, timestamp, 'checkpoints'), exist_ok=True)

        # Check if destination is provided
        if self.logging is not None:
            assert dst is not None

        # Initialize tensorboard writer (logging)
        if self.logging is not None:
            writer = SummaryWriter(log_dir=osp.join(dst, timestamp))

        # Parameterize optimizer
        optimizer = self.optimizer(model.parameters())

        # Pass optimizer to learning rate scheduler
        scheduler = self.scheduler(optimizer)

        # Initialize progressbar iterator
        tbar = trange(start_epoch, self.epochs, initial=start_epoch, total=self.epochs)

        for epoch in tbar:
            # Execute model training
            self.train_one_epoch(epoch, model, data_loader, optimizer, writer)

            # Execute model validation
            if val_loader is not None:
                result = self.validate_one_epoch(epoch, model, val_loader, writer)

            # Update learning rate
            scheduler.step()

            # Update progressbar
            tbar.set_postfix({k: float(v) for k, v in result.items()}, refresh=True)

            # Save checkpoint
            path = osp.join(dst, timestamp, 'checkpoints',
                            f"{timestamp}_checkpoint_{str(epoch).zfill(4)}.pt")
            torch.save(model, path)

        # Flush and close writer
        if self.logging is not None:
            writer.flush()
            writer.close()


def build_trainer(*args, **kwargs):
    return CentralizedTrainer.from_config(*args, **kwargs)
