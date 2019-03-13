# -*- coding: utf-8 -*-

import os
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from pandas.io.parsers import TextFileReader
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import Dataset


class LipidDataset(Dataset):
    """
    Lipid dataset class. Assuming genotypes and phenotypes are sorted properly
    and rows are index matched.
    """
    def __init__(self, root: str, train: bool, transforms = None) -> None:
        ohe: OneHotEncoder = OneHotEncoder()  # encoder
        self.transforms: Callable = torch.from_numpy()

        phenotype_dir: str = os.path.basename(root)
        input_file: str
        target_file: str
        if train:
            input_file = os.path.join(
                phenotype_dir, "train",
                "lipids_genotype_" + phenotype_dir + "_train.csv"
            )
            target_file = os.path.join(
                phenotype_dir, "train",
                "lipids_phenotype_" + phenotype_dir + "_train.csv"
            )

        else:
            input_file = os.path.join(
                phenotype_dir, "test",
                "lipids_genotype_" + phenotype_dir + "_test.csv"
            )
            target_file = os.path.join(
                phenotype_dir, "test",
                "lipids_phenotype_" + phenotype_dir + "_test.csv"
            )

        # One hot encoded input.
        # - 0 = homozygous ref
        # - 1 = heterozygous
        # - 2 = homozygous alt
        self.genotypes: torch.Tensor = self.transforms(
            ohe.fit_transform(
                pd.read_csv(input_file).drop("IID", axis = 1)
            ).todense()
        )

        # Target.
        self.phenotypes: torch.Tensor = self.transform(
            pd.read_csv(target_file, usecols = [1]).values
        )

        self.data_len: int = len(self.phenotypes.index)  # sample size
        self.feats: int = self.genotypes.shape[1]  # number of encoded features


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X: torch.Tensor = self.genotypes[idx]
        y: torch.Tensor = self.phenotypes[idx]
        return (X, y)


    def __len__(self) -> int:
        return self.data_len

