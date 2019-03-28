# -*- coding: utf-8 -*-

from typing import NoReturn, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel


class LipidMLP(BaseModel):
    def __init__(self, input_feats: int) -> None:
        super(LipidMLP, self).__init__()
        self.fc1 = nn.Linear(input_feats, 10000)
        self.fc2 = nn.Linear(10000, 300)
        self.fc3 = nn.Linear(300, 10)
        self.fc4 = nn.Linear(10, 1)
        # self.fc5 = nn.Linear(30, 3)
        # self.fc6 = nn.Linear(3, 1)
        self.dropout = nn.Dropout(0.2)


    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = inputs[0]
        x = F.relu(self.fc1(x))  # data is already flat, no need to flatten
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        # x = self.dropout(x)
        # x = F.relu(self.fc5(x))
        # x = self.dropout(x)
        # x = self.fc6(x)
        return x

