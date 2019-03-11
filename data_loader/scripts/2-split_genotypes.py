# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import pandas as pd

raw_genotype: str = sys.argv[1]
processed_dir: str = sys.argv[2]

chr_loc: pd.DataFrame = pd.read_csv(raw_genotype, usecols = ["CHR", "LOC"])

dir: str
for dir in os.listdir(processed_dir):
    id_list: str = os.path.join(
        processed_dir, dir, "id_list_" + dir + ".csv"
    )
    iids: pd.Series = pd.read_csv(id_list, squeeze = True)

    pt_iid_gt: pd.DataFrame = pd.read_csv(
        raw_genotype, usecols = iids.astype(str)
    )
    pt_iid_gt.columns: pd.Index = pt_iid_gt.columns.astype(int)

    output: pd.DataFrame = pd.concat(
        [chr_loc, pt_iid_gt.sort_index(axis = 1)], axis = 1
    )
    output.to_csv(
        os.path.join(processed_dir, dir, "lipids_genotype_" + dir + ".csv"),
        index = False,
        header = True
    )

