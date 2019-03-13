# -*- coding: utf-8 -*-

import torch


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        pred: torch.Tensor = torch.argmax(output, dim = 1)
        assert pred.shape[0] == len(target)
        correct: int = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    k: int = 3
) -> float:
    with torch.no_grad():
        pred: torch.Tensor = torch.topk(output, k, dim = 1)[1]
        assert pred.shape[0] == len(target)
        correct: int = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

