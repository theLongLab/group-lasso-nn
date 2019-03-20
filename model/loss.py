# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


def adj_rsqr(
    output: torch.Tensor,
    target: torch.Tensor,
    input_feats: int
) -> torch.Tensor:
    ss_res: torch.Tensor = torch.sum((target - output) ** 2)
    ss_tot: torch.Tensor = torch.sum((target - target.mean()) ** 2)
    rsquared: torch.Tensor = 1 - ss_res / ss_tot
    adj_rsqr: torch.Tensor = 1 - (1 - rsquared) * (len(target) - 1) \
        / (len(target) - input_feats - 1)
    return adj_rsqr

