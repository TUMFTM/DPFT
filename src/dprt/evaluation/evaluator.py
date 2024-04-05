from __future__ import annotations  # noqa: F407

from typing import Any, Callable, Dict, Iterable, List

import os.path as osp

import torch

from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dprt.models import load as load_model
from dprt.evaluation.exporters import build as build_exporter
from dprt.evaluation.metric import build_metric


class CentralizedEvaluator():
    def __init__(self,
                 metric: torch.nn.modules.loss._Loss = None,
                 exporter: Callable = None,
                 device: str = None,
                 logging: str = None,):
        """
        Arguments:
            logging: Logging frequency. One of either None,
                step or epoch.
        """
        # Initialize instance arrtibutes
        self.eval_fn = metric
        self.export_fn = exporter
        self.device = device
        self.logging = logging

    @classmethod
    def from_config(cls,
                    config: Dict[str, Any],
                    *args,
                    **kwargs) -> CentralizedEvaluator:  # noqa: F821
        metric = build_metric(
            config['evaluate']
        )
        exporter = build_exporter(config['evaluate']['exporter']['name'], config)
        device = torch.device(config['computing']['device'])
        logging = config['train'].get('logging')

        return cls(
            metric=metric,
            exporter=exporter,
            device=device,
            logging=logging
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.evaluate(*args, **kwargs)

    @staticmethod
    def _dict_to(data: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
        return {k: v.to(device) for k, v in data.items()}

    @staticmethod
    def log_scalars(writer, scalars: Dict[str, Any], epoch: int, prefix: str = None) -> None:
        # Get prefix
        prefix = f"{prefix}/" if prefix is not None else ""

        # Add scalar values
        for name, scalar in scalars.items():
            writer.add_scalar(prefix + name, scalar, epoch)

    @torch.no_grad()
    def evaluate_complexity(self, epoch: int, model: torch.nn.Module,
                            data_loader: Iterable, writer=None):
        # Set model to evaluation mode
        model.eval()

        # Get inference test input
        data, _ = next(iter(data_loader))

        # Load test data (to device)
        data: Dict[str, torch.Tensor] = self._dict_to(data, self.device)

        # Determine model complexity
        with get_accelerator().device(self.device):
            flops, macs, params = get_model_profile(
                model=model, args=(data,),
                print_profile=False, warm_up=10, as_string=False
            )

        # Log model complexity
        self.log_scalars(
            writer, {'FLOPS': flops, 'MACS': macs, 'Parameters': params},
            epoch, 'test'
        )

    @torch.no_grad()
    def evaluate_inference_time(self, epoch: int, model: torch.nn.Module,
                                data_loader: Iterable, writer=None):
        # Set model to evaluation mode
        model.eval()

        # Get inference test input
        data, _ = next(iter(data_loader))

        # Load test data (to device)
        data: Dict[str, torch.Tensor] = self._dict_to(data, self.device)

        # Initialize loggers
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings = torch.zeros((repetitions, 1))

        # GPU warm-up
        for _ in range(10):
            model(data)

        # Measure performance
        for rep in range(repetitions):
            starter.record()
            model(data)
            ender.record()
            # Wait for GPU sync
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

        # Calculate mean and std inference time in milliseconds
        mean_syn = torch.sum(timings) / repetitions
        std_syn = torch.std(timings)

        # Log inference time measures
        self.log_scalars(
            writer, {'Inference_time_mean_ms': mean_syn, 'Inference_time_std_ms': std_syn},
            epoch, 'test'
        )

    @torch.no_grad()
    def evaluate_one_epoch(self, epoch: int, model: torch.nn.Module,
                           data_loader: Iterable, writer=None, dst: str = None):
        # Set model to evaluation mode
        model.eval()

        # Initialize epoch logs
        scalars = {}

        with tqdm(total=len(data_loader)) as pbar:
            for i, (data, labels) in enumerate(data_loader):
                # Load data and labels (to device)
                labels: List[Dict[str, torch.Tensor]] = \
                    [self._dict_to(label, self.device) for label in labels]
                data: Dict[str, torch.Tensor] = \
                    self._dict_to(data, self.device)

                # Make prediction
                output = model(data)

                # Evaluate model output
                metrics = self.eval_fn(output, labels)

                # Log evaluation step
                if self.logging == 'step':
                    self.log_scalars(writer, metrics, i + epoch * len(data_loader), 'test')

                # Add values to epoch log
                if self.logging == 'epoch':
                    for k, v in metrics.items():
                        scalars[k] = scalars.get(k, 0) + v

                # Export predictions
                if self.export_fn is not None:
                    self.export_fn(output, labels, i * len(labels), dst)

                # Report training progress
                pbar.update()

        if self.logging == 'epoch':
            # Average epoch logs
            scalars = {k: v / (i + 1) for k, v in scalars.items()}

            # Write epoch logs
            self.log_scalars(writer, scalars, epoch, 'test')

    def evaluate(self, checkpoint: str, data_loader: Iterable, dst: str = None):
        # Load model from checkpoint
        model, epoch, timestamp = load_model(checkpoint)

        # Load model (to device)
        model.to(self.device)

        # Check if destination is provided
        if self.logging is not None:
            dst = osp.join(dst, timestamp)

        # Initialize tensorboard writer (logging)
        if self.logging is not None:
            writer = SummaryWriter(log_dir=dst)

        # Evaluate model performance
        self.evaluate_one_epoch(epoch, model, data_loader, writer, dst)

        # Evaluate model inference time
        self.evaluate_inference_time(epoch, model, data_loader, writer)

        # Evaluate model complexity
        self.evaluate_complexity(epoch, model, data_loader, writer)

        # Flush and close writer
        if self.logging is not None:
            writer.flush()
            writer.close()


def build_evaluator(*args, **kwargs):
    return CentralizedEvaluator.from_config(*args, **kwargs)
