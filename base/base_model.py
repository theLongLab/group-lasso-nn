# -*- coding: utf-8 -*-

import logging
from typing import Any, Iterable, NoReturn, Tuple, Union

import numpy as np
import torch
from torch.nn import Module


class BaseModel(Module):
    """
    Base class for all models.
    Implements logging and NotImplementedError for forward pass function.
    """
    def __init__(self) -> None:
        super(BaseModel, self).__init__()
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)


    def forward(self, *inputs: torch.Tensor) -> Union[NoReturn, torch.Tensor]:
        """
        Forward pass function. If not implemented in child class, raise
        `NotImplementedError`.:

        Parameters
        ----------
        *input : torch.Tensor
            Transformed input data.

        Returns
        -------
        torch.Tensor
            Model output.
        """
        raise NotImplementedError


    def summary(self) -> None:
        """
        Print model summary in logger.
        """
        # Model parameters requiring gradient computation.
        model_parameters: Iterable[torch.Tensor] = filter(
            lambda p: p.requires_grad, self.parameters()
        )

        # Total number of tuneable parameters (sum of model parameter * number
        # of samples for each model parameter)
        n_params: int = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info("Trainable parameters: {}".format(n_params))
        self.logger.info(self)


    def __str__(self) -> str:
        """
        Prints model summary in stdout.

        Returns
        -------
        str
            The model representation and the number of trainable parameters.
        """
        model_parameters: Iterable[torch.Tensor] = filter(
            lambda p: p.requires_grad, self.parameters()
        )

        n_params: int = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() \
            + "\nTrainable parameters: {}".format(n_params)

