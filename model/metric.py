# -*- coding: utf-8 -*-

import torch


def my_metric(output: torch.Tensor, target: torch.Tensor) -> float:
    """
    Metric 1: highest
    """
    with torch.no_grad():
        pred: torch.Tensor = torch.argmax(output, dim = 1)
        assert pred.shape[0] == len(target)
        correct: int = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def my_metric2(output: torch.Tensor, target: torch.Tensor, k: int = 3) -> float:
    with torch.no_grad():
        pred: torch.Tensor = torch.topk(output, k, dim = 1)[1]
        assert pred.shape[0] == len(target)
        correct: int = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

