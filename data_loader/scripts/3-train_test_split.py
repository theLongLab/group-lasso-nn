# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
import torch


# Input params
genotype_file: str = sys.argv[1]
phenotype_file: str = sys.argv[2]
output_root: str = sys.argv[3]

# Files
genotypes: np.ndarray = np.loadtxt(genotype_file, delimiter = ',', skiprows = 1)
phenotypes: np.ndarray = np.loadtxt(
    phenotype_file, delimiter = ',', skiprows = 1
)

# Folder prep
pt: str = os.path.basename(output_root)
train_dir: str = os.path.join(output_root, "train")
test_dir: str = os.path.join(output_root, "test")
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)

# Split
gt_train: np.ndarray
gt_test: np.ndarray
pt_train: np.ndarray
pt_test: np.ndarray
gt_train, gt_test, pt_train, pt_test = train_test_split(
    genotypes, phenotypes, test_size = 0.2
)
print("split done")

# Save
torch.save(
    torch.from_numpy(gt_train).float(),
    os.path.join(train_dir, "lipids_genotype_" + pt + "_train.pth")
)
print("gt train done")

torch.save(
    torch.from_numpy(gt_test).float(),
    os.path.join(test_dir, "lipids_genotype_" + pt + "_test.pth")
)
print("gt test done")

torch.save(
    torch.from_numpy(pt_train).float(),
    os.path.join(train_dir, "lipids_phenotype_" + pt + "_train.pth")
)
print("pt train done")

torch.save(
    torch.from_numpy(pt_test).float(),
    os.path.join(test_dir, "lipids_phenotype_" + pt + "_test.pth")
)
print("pt test done")

