# -*- coding: utf-8 -*-

import os
import sys
from typing import Tuple

import pandas as pd
from pandas.io.parsers import TextFileReader
from sklearn.model_selection import train_test_split

genotype_file: str = sys.argv[1]
phenotype_file: str = sys.argv[2]
output_root: str = sys.argv[3]

genotypes: TextFileReader = pd.read_csv(
    genotype_file, index_col = 0, chunksize = 100
)
phenotypes: pd.DataFrame = pd.read_csv(phenotype_file, index_col = 0)

os.mkdir(os.path.join(output_root, "train"))
os.mkdir(os.path.join(output_root, "test"))
pt: str = os.path.basename(output_root)

chunk_idx: int
gt_chunk: pd.DataFrame
for chunk_idx, gt_chunk in enumerate(genotypes):
    print("\ncurr chunk: {}".format(chunk_idx))
    file_mode: str
    header_write: bool
    if chunk_idx == 0:
        file_mode = 'w'
        header_write = True
    else:
        file_mode = 'a'
        header_write = False

    gt_test: pd.DataFrame
    pt_train: pd.DataFrame
    pt_test: pd.DataFrame
    gt_train, gt_test, pt_train, pt_test = train_test_split(
        gt_chunk, phenotypes.loc[gt_chunk.index.values], test_size = 0.2
    )
    print("split done")

    gt_train.to_csv(
        os.path.join(
            output_root, "train", "lipids_genotype_" + pt + "_train.csv"
        ),
        header = header_write,
        mode = file_mode
    )
    print("gt train done")

    gt_test.to_csv(
        os.path.join(
            output_root, "test", "lipids_genotype_" + pt + "_test.csv"
        ),
        header = header_write,
        mode = file_mode
    )
    print("gt test done")

    pt_train.to_csv(
        os.path.join(
            output_root, "train", "lipids_phenotype_" + pt + "_train.csv"
        ),
        header = header_write,
        mode = file_mode
    )
    print("pt train done")

    pt_test.to_csv(
        os.path.join(
            output_root, "test", "lipids_phenotype_" + pt + "_test.csv"
        ),
        header = header_write,
        mode = file_mode
    )
    print("pt test done")

