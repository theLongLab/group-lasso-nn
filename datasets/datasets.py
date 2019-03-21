# -*- coding: utf-8 -*-

import os
from typing import Callable, Optional, Tuple, Union

import numpy as np
import dask.dataframe as dd
import torch
from torch.utils.data import Dataset


class LipidDataset(Dataset):
    """
    Lipid dataset class. Assuming genotypes and phenotypes are sorted properly
    and rows are index matched.
    """
    def __init__(
        self,
        root: str,
        train: bool,
        block_size: Optional[float],
        data_dtype: Union[str, type, None],
        dask_sample: Optional[int],
        transforms = None
    ) -> None:
        self.transforms: Callable = torch.from_numpy  # transformation fn

        # Data folder.
        phenotype_dir: str = os.path.basename(root)
        train_test: str = "test"
        if train:
            train_test = "train"

        # File paths.
        input_file: str = os.path.join(
            root, train_test,
            "lipids_genotype_" + phenotype_dir + '_' + train_test + ".csv"
        )
        target_file: str = os.path.join(
            root, train_test,
            "lipids_phenotype_" + phenotype_dir + '_' + train_test + ".csv"
        )

        # Inputs.
        self.genotypes: torch.Tensor = self.transforms(
            np.array(
                dd.read_csv(
                    input_file, blocksize = block_size
                ).drop("IID", axis = 1).values
            )
        ).float()
        print("Input loading complete.")

        # Target.
        self.phenotypes: torch.Tensor = self.transforms(
            np.array(
                dd.read_csv(
                    target_file, usecols = [1], blocksize = block_size
                ).values
            )
        ).float()
        print("Target loading complete.")

        self.data_len: int = self.genotypes.shape[0]  # sample size
        self.feats: int = self.genotypes.shape[1]  # number of encoded features


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X: torch.Tensor = self.genotypes[idx]
        y: torch.Tensor = self.phenotypes[idx]
        return (X, y)


    def __len__(self) -> int:
        return self.data_len

