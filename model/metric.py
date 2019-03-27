# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


def rmse(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.sqrt(F.mse_loss(output, target))


def mae(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return F.l1_loss(output, target)


def corr(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        vx: torch.Tensor = output - torch.mean(output)
        vy: torch.Tensor = target - torch.mean(target)
        return torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))


def rsquared(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        ss_res: torch.Tensor = torch.sum((target - output) ** 2)
        ss_tot: torch.Tensor = torch.sum((target - target.mean()) ** 2)
        rsquared: torch.Tensor = 1 - ss_res / ss_tot
        return rsquared


def adj_rsqr(
    output: torch.Tensor,
    target: torch.Tensor,
    input_feats: int
) -> torch.Tensor:
    with torch.no_grad():
        ss_res: torch.Tensor = torch.sum((target - output) ** 2)
        ss_tot: torch.Tensor = torch.sum((target - target.mean()) ** 2)
        rsquared: torch.Tensor = 1 - ss_res / ss_tot
        adj_rsqr: torch.Tensor = 1 - (1 - rsquared) * (len(target) - 1) \
            / (len(target) - input_feats - 1)
        return adj_rsqr

