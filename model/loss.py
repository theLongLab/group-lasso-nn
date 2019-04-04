# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


def rmse(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(F.mse_loss(output, target))

