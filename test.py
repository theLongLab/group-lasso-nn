# -*- coding: utf-8 -*-

import argparse
import os
from typing import Any, Callable, Iterable, List, Optional

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from train import get_instance


def main(config: dict, resume: Optional[str]):
    # Instantiate data loader.
    data_loader: DataLoader = getattr(
        module_data,
        config["data_loader"]["type"]
    )(
        config["data_loader"]["args"]["data_dir"],
        batch_size = 512,
        shuffle = False,
        validation_split = 0.0,
        training = False,
        num_workers = 2
    )

    # Instantiate model and print summary.
    model: Module = get_instance(
        module_arch, "arch", config, data_loader.dataset.feats
    )
    model.summary()

    # Obtain function handles of loss and metrics.
    loss_fn: Callable = getattr(module_loss, config["loss"])
    loss_args: dict = config["loss"]["args"]
    metric_fns: List[Callable] = [
        getattr(module_metric, met) for met in config["metrics"]
    ]
    metric_args: List[dict] = [
        config["metrics"][met] for met in config["metrics"]
    ]

    # Load state dict.
    checkpoint: dict = torch.load(resume)
    state_dict: dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # Prepare model for testing.
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    total_loss: float = 0.0
    total_metrics: torch.Tensor = torch.zeros(len(metric_fns))

    with torch.no_grad():
        i: int
        data: torch.Tensor
        target: torch.Tensor
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output: torch.Tensor = model(data)
            #
            # save sample images, or do something with output here
            #
            # computing loss, metrics on test set
            loss: torch.Tensor = loss_fn(output, target, **loss_args)
            batch_size: int = data.shape[0]
            total_loss += loss.item() * batch_size

            j: int
            metric: Callable
            for j, metric in enumerate(metric_fns):
                total_metrics[j] += metric(output, target, **metric_args[j]) \
                    * batch_size

    n_samples: int = len(data_loader.sampler)
    log: dict = {"loss": total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item()
        / n_samples for i, met in enumerate(metric_fns)
    })
    print(log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Michael's adapted PyTorch Template -- Testing")

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

    if args.resume:
        config = torch.load(args.resume)["config"]

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)

