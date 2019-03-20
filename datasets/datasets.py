# -*- coding: utf-8 -*-

import os
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.io.parsers import TextFileReader
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
        chunksize: Optional[int],
        transforms = None
    ) -> None:
        self.transforms: Callable = torch.from_numpy

        phenotype_dir: str = os.path.basename(root)
        train_test: str = "test"
        if train:
            train_test = "train"

        input_file = os.path.join(
            root, train_test,
            "lipids_genotype_" + phenotype_dir + '_' + train_test + ".csv"
        )
        target_file = os.path.join(
            root, train_test,
            "lipids_phenotype_" + phenotype_dir + '_' + train_test + ".csv"
        )

        self.genotypes: torch.Tensor
        gt: Union[pd.DataFrame, TextFileReader] = pd.read_csv(
            input_file, chunksize = chunksize
        )

        if chunksize is None:
            self.genotypes = self.transforms(gt.drop("IID", axis = 1)).float()

        else:
            print("Reading input chunks...")
            gt_chunk: pd.DataFrame
            chunk_idx: int = 0
            for gt_chunk in gt:
                print("Current chunk: {}".format(chunk_idx))
                gt_chunk_tensor: torch.Tensor = self.transforms(
                    gt_chunk.drop("IID", axis = 1)
                ).float()
                print("Chunk {} encoding complete.".format(chunk_idx))
                del gt_chunk  # mem management

                try:
                    self.genotypes = torch.cat([
                        self.genotypes, gt_chunk_tensor
                    ])
                except AttributeError:
                    self.genotypes = gt_chunk_tensor

                del gt_chunk_tensor
                print("Chunk {} complete.".format(chunk_idx))

        del gt
        print("Input loading complete.")

        # Target.
        self.phenotypes: torch.Tensor = self.transforms(
            pd.read_csv(target_file, usecols = [1]).values
        ).float()

        self.data_len: int = self.genotypes.shape[0]  # sample size
        self.feats: int = self.genotypes.shape[1]  # number of encoded features


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X: torch.Tensor = self.genotypes[idx]
        y: torch.Tensor = self.phenotypes[idx]
        return (X, y)


    def __len__(self) -> int:
        return self.data_len

