# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


# This file is for adapting loss functions as callables that can be reached
# with a single string. Below is an example with negative log-likelihood loss.
def nll_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.nll_loss(output, target)

