# -*- coding: utf-8 -*-

from typing import NoReturn, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel


class LipidMLP_LR(BaseModel):
    def __init__(self, input_feats: int) -> None:
        super(LipidMLP_LR, self).__init__()
        self.output = nn.Linear(input_feats, 1)


    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = inputs[0]
        return self.output(x)


class LipidMLP_Shallow(BaseModel):
    def __init__(self, input_feats: int) -> None:
        super(LipidMLP_Shallow, self).__init__()
        self.dense1 = nn.Linear(input_feats, 100)
        self.output = nn.Linear(100, 1)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(0.2)


    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = inputs[0]
        x = self.prelu(self.fc1(x))  # data is already flat, no need to flatten
        x = self.dropout(x)
        return self.output(x)


class LipidMLP_Deep(BaseModel):
    def __init__(self, input_feats: int) -> None:
        super(LipidMLP_Deep, self).__init__()
        self.dense1 = nn.Linear(input_feats, 1000)
        self.dense2 = nn.Linear(1000, 100)
        self.dense3 = nn.Linear(100, 10)
        self.output = nn.Linear(10, 1)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(0.2)


    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = inputs[0]
        x = self.prelu(self.fc1(x))  # data is already flat, no need to flatten
        x = self.dropout(x)
        x = self.prelu(self.fc2(x))
        x = self.dropout(x)
        x = self.prelu(self.fc3(x))
        x = self.dropout(x)
        return self.output(x)

