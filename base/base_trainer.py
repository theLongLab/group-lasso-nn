# -*- coding: utf-8 -*-

import datetime
import json
import logging
import math
import os
from typing import Callable, List, NoReturn, Optional, Tuple, Union

import torch
from torch.nn import DataParallel, Module
from torch.optim import Optimizer

from utils.logger import Logger
from utils.util import ensure_dir
from utils.visualization import WriterTensorboardX


class BaseTrainer:
    """
    Base class for all trainers.
    """
    def __init__(
        self,
        model: Module,
        loss: Callable,
        loss_args: dict,
        metrics: List[Callable],
        metric_args: List[dict],
        optimizer: Optimizer,
        config: dict,
        resume: Optional[str] = None,
        train_logger: Optional[Logger] = None
    ) -> None:
        self.config: dict = config
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)

        # Setup GPU device if available.
        self.device: str
        device_ids: List[int]
        self.device, device_ids = self._prepare_device(config["n_gpu"])

        # Move model into device(s).
        self.model: Module = model.to(self.device)
        if len(device_ids) > 1:
            self.model: Module = DataParallel(model, device_ids = device_ids)

        self.loss: Callable = loss
        self.loss_args: dict = loss_args
        self.metrics: List[Callable] = metrics
        self.metric_args: List[dict] = metric_args
        self.optimizer: Optimizer = optimizer
        self.train_logger: Optional[Logger] = train_logger

        cfg_trainer: dict = config["trainer"]
        self.epochs: int = cfg_trainer["epochs"]
        self.save_period: int = cfg_trainer["save_period"]
        self.verbosity: int = cfg_trainer["verbosity"]
        self.monitor: str = cfg_trainer.get("monitor", "off")

        self.mnt_mode: str
        self.mnt_best: float

        # Configuration to monitor model performance and save the best result.
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0

        else:
            self.mnt_mode: str
            self.mnt_metric: str
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = math.inf if self.mnt_mode == "min" \
                else -math.inf
            self.early_stop: float = cfg_trainer.get("early_stop", math.inf)

        self.start_epoch = 1

        # Setup directory for saving checkpoints.
        start_time: str = datetime.datetime.now().strftime("%m%d_%H%M%S")
        self.checkpoint_dir: str = os.path.join(
            cfg_trainer["save_dir"], config["name"], start_time
        )

        # Setup visualization writer instance.
        writer_dir: str = os.path.join(
            cfg_trainer["log_dir"], config["name"], start_time
        )
        self.writer: WriterTensorboardX = WriterTensorboardX(
            writer_dir, self.logger, cfg_trainer["tensorboardX"]
        )

        # Save configuration file into checkpoint directory.
        ensure_dir(self.checkpoint_dir)
        config_save_path: str = os.path.join(
            self.checkpoint_dir, "config.json"
        )
        with open(config_save_path, 'w') as handle:
            json.dump(config, handle, indent = 4, sort_keys = False)

        if resume:
            self._resume_checkpoint(resume)


    def _prepare_device(self, n_gpu_use: int) -> Tuple[torch.device, List[int]]:
        """
        Setup GPU device if available and move model into configured device.

        Parameters
        ----------
        n_gpu_use : int
            Number of GPUs to use.

        Returns
        -------
        tuple
            A tuple of the device in use and a list of device ids.
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There's no GPU available on this machine, training "
                + "will be performed on CPU."
            )
            n_gpu_use = 0

        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU's configured to use is "
                + "{}, but only {} ".format(n_gpu_use, n_gpu)
                + "are available on this machine."
            )
            n_gpu_use = n_gpu

        device: torch.device = torch.device(
            "cuda:0" if n_gpu_use > 0 else "cpu"
        )
        list_ids: List[int] = list(range(n_gpu_use))

        return device, list_ids


    def train(self) -> None:
        """
        Full training logic.
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            result: dict = self._train_epoch(epoch)

            # Save logged informations into log dict.
            log: dict = {"epoch": epoch}
            for key, value in result.items():
                if key == "metrics":
                    log.update({
                        metric.__name__: value[i] for i, metric in
                        enumerate(self.metrics)
                    })

                elif key == "val_metrics":
                    log.update({
                        "val_" + metric.__name__: value[i] for i, metric in
                        enumerate(self.metrics)
                    })

                else:
                    log[key] = value

            # Print logged informations to the screen.
            if self.train_logger is not None:
                self.train_logger.add_entry(log)
                if self.verbosity >= 1:
                    for key, value in log.items():
                        self.logger.info(
                            "    {:15s}: {}".format(str(key), value)
                        )

            # Evaluate model performance according to configured metric,
            # save best checkpoint as model_best.
            best: bool = False
            not_improved_count: int
            if self.mnt_mode != "off":
                try:
                    # Check whether model performance has improved or not,
                    # according to specified metric(mnt_metric).
                    improved: bool \
                        = (
                            self.mnt_mode == "min"
                            and log[self.mnt_metric] < self.mnt_best
                        ) or (
                            self.mnt_mode == "max"
                            and log[self.mnt_metric] > self.mnt_best
                        )

                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found.".format(
                            self.mnt_metric
                        ) + "Model performance monitoring is disabled."
                    )
                    self.mnt_mode = "off"
                    improved = False
                    not_improved_count = 0

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True

                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for "
                        + "{} epochs. ".format(self.early_stop)
                        + "Training stops."
                    )
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch = epoch, save_best = best)


    def _train_epoch(self, epoch: int) -> Union[NoReturn, dict]:
        """
        Training logic for an epoch. If not implemented in child class, raise
        NotImplementedError

        Parameters
        ----------
        epoch : int
            The current epoch number.

        Returns
        -------
        dict
            Logged info in a dictionary.
        """
        raise NotImplementedError


    def _save_checkpoint(self, epoch: int, save_best: bool = False) -> None:
        """
        Saving current state as checkpoint.

        Parameters
        ----------
        epoch : int
            The current epoch number.

        save_best: bool
            If True, rename the saved checkpoint to "model_best.pth"
        """
        arch: str = type(self.model).__name__
        state: dict = {
            "arch": arch,
            "epoch": epoch,
            "logger": self.train_logger,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config
        }

        filename: str = os.path.join(
            self.checkpoint_dir, "checkpoint-epoch{}.pth".format(epoch)
        )
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

        if save_best:
            best_path: str = os.path.join(self.checkpoint_dir, "model_best.pth")
            torch.save(state, best_path)
            self.logger.info(
                "Saving current best: {} ...".format("model_best.pth")
            )


    def _resume_checkpoint(self, resume_path: str) -> None:
        """
        Resume from a specified checkpoint.

        Parameters
        ----------
        resume_path : str
            File path to the checkpoint file.
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint: dict = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # Load architecture parameters from checkpoint.
        if checkpoint["config"]["arch"] != self.config["arch"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is "
                + "different from that of checkpoint. This may yield an "
                + "exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # Load optimizer state from checkpoint only when optimizer type has not
        # changed.
        if checkpoint["config"]["optimizer"]["type"] != \
                self.config["optimizer"]["type"]:
            self.logger.warning(
                "Warning: Optimizer type given in config file is different "
                + "from that of checkpoint. Optimizer parameters not being "
                + "resumed."
            )

        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.train_logger = checkpoint["logger"]
        self.logger.info(
            "Checkpoint '{}' (epoch {}) loaded".format(
                resume_path, self.start_epoch
            )
        )

