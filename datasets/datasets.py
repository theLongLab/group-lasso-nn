# -*- coding: utf-8 -*-

import sys
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import Dataset


class LipidDataset(Dataset):
    """
    Lipid dataset class. Assuming genotypes and phenotypes are sorted properly
    and rows are index matched.
    """
    def __init__(
        self,
        genotype_file: str,
        phenotype_file: str,
        transforms = None
    ) -> None:
        ohe: OneHotEncoder = OneHotEncoder()  # encoder
        self.transforms: Callable = torch.from_numpy()

        # One hot encoded input.
        # - 0 = homozygous ref
        # - 1 = heterozygous
        # - 2 = homozygous alt
        self.genotypes: torch.Tensor = self.transforms(
            ohe.fit_transform(
                pd.read_csv(genotype_file).drop("IID", axis = 1)
            ).todense()
        )

        # Target.
        self.phenotypes: torch.Tensor = self.transform(
            pd.read_csv(phenotype_file, usecols = [1]).values
        )

        self.data_len: int = len(self.phenotypes.index)  # sample size


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X: torch.Tensor = self.genotypes[idx]
        y: torch.Tensor = self.phenotypes[idx]
        return (X, y)


    def __len__(self) -> int:
        return self.data_len

