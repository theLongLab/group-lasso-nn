# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


def mae(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(output, target)

