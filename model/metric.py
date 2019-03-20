# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


def rmse(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.sqrt(F.mse_loss(output, target))


def mae(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return F.l1_loss(output, target)


def rsquared(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        ss_res: torch.Tensor = torch.sum((target - output) ** 2)
        ss_tot: torch.Tensor = torch.sum((target - target.mean()) ** 2)
        rsquared: torch.Tensor = 1 - ss_res / ss_tot
        return rsquared


def corr(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.sqrt(rsquared(output, target))

