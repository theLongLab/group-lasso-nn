# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import pandas as pd

raw_genotype: str = sys.argv[1]
processed_dir: str = sys.argv[2]

chr_loc: pd.DataFrame = pd.read_csv(raw_genotype, usecols = ["CHR", "LOC"])
snp: pd.DataFrame = chr_loc.CHR.astype(str) + '_' + chr_loc.LOC.astype(str)
del chr_loc

pt: str = os.path.basename(processed_dir)
id_list: str = os.path.join(processed_dir, "id_list_" + pt + ".csv")
iids: pd.Series = pd.read_csv(id_list, squeeze = True)

pt_iid_gt: pd.DataFrame = pd.read_csv(raw_genotype, usecols = iids.astype(str))
pt_iid_gt.columns: pd.Index = pt_iid_gt.columns.astype(int)

output_path = os.path.join(processed_dir, "lipids_genotype_" + pt + ".csv")
output: pd = pd.concat(
    [snp, pt_iid_gt.sort_index(axis = 1)], axis = 1
).transpose()
output.to_csv(index = True, index_label = "IID", header = False)

