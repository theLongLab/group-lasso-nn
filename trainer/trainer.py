# -*- coding: utf-8 -*-

from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from base import BaseTrainer
from utils.logger import Logger


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(
        self,
        model: Module,
        loss: Callable,
        metrics: List[Callable],
        optimizer: Optimizer,
        config: dict,
        resume: Optional[str],
        data_loader: DataLoader,
        valid_data_loader: Optional[DataLoader] = None,
        lr_scheduler = None,
        train_logger: Optional[Logger] = None
    ) -> None:
        super(Trainer, self).__init__(
            model, loss, metrics, optimizer, config, resume, train_logger
        )
        self.config: dict = config
        self.data_loader: DataLoader = data_loader
        self.valid_data_loader: Optional[DataLoader] = valid_data_loader
        self.do_validation: bool = self.valid_data_loader is not None

        self.lr_scheduler = lr_scheduler
        self.log_step: int = int(np.sqrt(data_loader.batch_size))


    def _eval_metrics(
        self,
        output: torch.Tensor,
        target: torch.Tensor
    ) -> np.ndarray:
        acc_metrics: np.ndarray = np.zeros(len(self.metrics))

        i: int
        metric: Callable
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar(f"{metric.__name__}", acc_metrics[i])
        return acc_metrics


    def _train_epoch(self, epoch: int) -> dict:
        """
        Training logic for a single epoch.

        Parameters
        ----------
        epoch : int
            Current epoch.

        Returns
        -------
        dict
            Logged info.

        Notes:
        - If you have additional information to record, for example:
            > additional_log = {"x": x, "y": y}
          you have to merge it with log before return. i.e.
            > log = {**log, **additional_log}
            > return log
        - The metrics in log must have the key "metrics".
        """
        self.model.train()

        total_loss: float = 0
        total_metrics: np.ndarray = np.zeros(len(self.metrics))

        batch_idx: int
        data: torch.Tensor
        target: torch.Tensor
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output: torch.Tensor = self.model(data)

            # for adjusted r-squared
            loss: torch.Tensor = self.loss(output, target, data.shape[1])
            loss.backward()
            self.optimizer.step()

            self.writer.set_step(
                (epoch - 1) * len(self.data_loader) + batch_idx
            )
            self.writer.add_scalar("loss", loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, target)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        self.data_loader.n_samples,
                        100.0 * batch_idx / len(self.data_loader),
                        loss.item()
                    )
                )
                self.writer.add_image(
                    "input", make_grid(data.cpu(), nrow = 8, normalize = True)
                )

        log: dict = {
            "loss": total_loss / len(self.data_loader),
            "metrics": (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log: dict = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log


    def _valid_epoch(self, epoch: int) -> dict:
        """
        Validation after training an epoch.

        Parameters
        ----------
        epoch : int
            Current epoch.

        Returns
        -------
        dict
            Logged info.

        Notes:
        - The validation metrics in log must have the key "val_metrics".
        """
        self.model.eval()
        total_val_loss: int = 0
        total_val_metrics: np.ndarray = np.zeros(len(self.metrics))

        with torch.no_grad():
            # This function is only called if validation is being performed,
            # so `self.valid_data_loader` must be a `DataLoader` object and not
            # `None`.
            self.valid_data_loader: DataLoader
            batch_idx: int
            data: torch.Tensor
            target: torch.Tensor
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output: torch.Tensor = self.model(data)
                loss: torch.Tensor = self.loss(output, target)

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader)
                    + batch_idx, "valid"
                )
                self.writer.add_scalar("loss", loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)
                self.writer.add_image(
                    "input", make_grid(data.cpu(), nrow = 8, normalize = True)
                )

        return {
            "val_loss": total_val_loss / len(self.valid_data_loader),
            "val_metrics": (
                total_val_metrics / len(self.valid_data_loader)
            ).tolist()
        }

