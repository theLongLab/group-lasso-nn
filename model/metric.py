# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


def mse(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return F.mse_loss(output, target)

