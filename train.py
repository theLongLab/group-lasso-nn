# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Callable, Iterable, List, Optional

import adabound
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer import Trainer
from utils import Logger


def get_instance(module, name: str, config: dict, *args) -> Any:
    return getattr(module, config[name]["type"])(*args, **config[name]["args"])


def main(config: dict, resume: Optional[str]) -> None:
    train_logger: Logger = Logger()

    # Instantiate data loaders.
    data_loader: DataLoader = get_instance(module_data, "data_loader", config)
    valid_data_loader: Optional[DataLoader] = data_loader.split_validation()

    # Instantiate model.
    model: Module = get_instance(
        module_arch, "arch", config, data_loader.dataset.feats
    )
    print(model)

    # Obtain function handles for loss and metrics.
    loss: Callable = getattr(module_loss, config["loss"]["type"])
    loss_args: dict = config["loss"]["args"]
    metrics: List[Callable] = [
        getattr(module_metric, met) for met in config["metrics"]
    ]
    metric_args: List[dict] = [
        config["metrics"][met] for met in config["metrics"]
    ]

    # Instantiate optimizer and learning rate scheduler.
    # Delete every line containing lr_scheduler to disable scheduler.
    trainable_params: Iterable[torch.Tensor] = filter(
        lambda p: p.requires_grad, model.parameters()
    )

    module_optim = ''
    if config["type"] == "Adabound":
        module_optim = adabound
    else:
        module_optim = torch.optim

    optimizer = get_instance(
        module_optim, "optimizer", config, trainable_params
    )
    lr_scheduler = get_instance(
        torch.optim.lr_scheduler, "lr_scheduler", config, optimizer
    )

    # Instantiate trainer.
    trainer: Trainer = Trainer(
        model = model,
        loss = loss,
        loss_args = loss_args,
        metrics = metrics,
        metric_args = metric_args,
        optimizer = optimizer,
        config = config,
        resume = resume,
        data_loader = data_loader,
        valid_data_loader = valid_data_loader,
        lr_scheduler = lr_scheduler,
        train_logger = train_logger
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Michael's adapted PyTorch Template -- Training"
    )

    # Not specified as required here as a custom error message is raised below.
    parser.add_argument(
        "-c",
        "--config",
        default = None,
        type = str,
        help = "Config JSON file path."
    )

    parser.add_argument(
        "-r",
        "--resume",
        default = None,
        type = str,
        help = "Latest PTH checkpoint file path (default: None)"
    )

    parser.add_argument(
        "-d",
        "--device",
        default = None,
        type = str,
        help = "Indices of GPUs to enable (default: all)"
    )

    args = parser.parse_args()

    if args.config:
        # Load config file.
        config: dict = json.load(open(args.config))
        path: str = os.path.join(config["trainer"]["save_dir"], config["name"])

    elif args.resume:
        # Load config file from checkpoint, in case a new config file is not
        # given.

        # Use "--config" and "--resume" arguments together to load trained model
        # and train more with changed config.
        config = torch.load(args.resume)["config"]

    else:
        raise AssertionError(
            "Configuration file need to be specified. Add '-c config.json', for"
            + " example."
        )

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)

