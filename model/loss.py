# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


def rsquared_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ss_res: torch.Tensor = torch.sum((target - output) ** 2)
    ss_tot: torch.Tensor = torch.sum((target - target.mean()) ** 2)
    rsquared: torch.Tensor = 1 - ss_res / ss_tot
    return -rsquared

