# -*- coding: utf-8 -*-

import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


genotype_file = sys.argv[1]
phenotype_file = sys.argv[2]



class LipidGenotypeDataset(Dataset):
    """
    genotype
    """
    def __init__(
        self,
        file_path: str,
        transform = None
    ) -> None:
        pass

    def __len__(self):
        return None

    def __getitem__(self, idx):
        return None







