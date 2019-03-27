# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


def huber(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(output, target)

