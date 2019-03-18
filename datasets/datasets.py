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
    def __init__(
        self,
        root: str,
        train: bool,
        chunksize: int,
        transforms = None
    ) -> None:
        ohe: OneHotEncoder = OneHotEncoder(categories = "auto")  # encoder
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

        # One hot encoded input.
        # - 0 = homozygous ref
        # - 1 = heterozygous
        # - 2 = homozygous alt
        self.genotypes: torch.Tensor
        ohe_dummy_rows: pd.DataFrame
        num_cols: int
        genotypes: TextFileReader = pd.read_csv(
            input_file, chunksize = chunksize
        )

        print("Reading input chunks...")
        gt_chunk: pd.DataFrame
        chunk_idx: int
        for chunk_idx, gt_chunk in enumerate(genotypes):
            print("Current chunk: {}".format(chunk_idx))
            if chunk_idx == 0:
                print("Initializing dummies...")
                num_cols = len(gt_chunk.columns) - 1  # not counting IID col
                dummy_row: np.ndarray = np.zeros(num_cols, dtype = int)
                ohe_dummy_rows = pd.DataFrame(
                    columns = gt_chunk.columns[1:],
                    data = np.array([dummy_row, dummy_row + 1, dummy_row + 2])
                )

            gt_chunk = pd.concat(
                [gt_chunk.drop("IID", axis = 1), ohe_dummy_rows]
            )
            print("Chunk {} dummying complete.".format(chunk_idx))

            gt_chunk_tensor: torch.Tensor = self.transforms(
                ohe.fit_transform(gt_chunk).todense()[:-3, :]  # exclude dummies
            ).float()
            print("Chunk {} encoding complete.".format(chunk_idx))

            try:
                self.genotypes = torch.cat([self.genotypes, gt_chunk_tensor])
            except AttributeError:
                self.genotypes = gt_chunk_tensor

            print("Chunk {} complete.".format(chunk_idx))

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
