# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


def corr(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    vx: torch.Tensor = output - torch.mean(output)
    vy: torch.Tnsor = target - torch.mean(target)
    return torch.sum(vx * vy) / (
        torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))




