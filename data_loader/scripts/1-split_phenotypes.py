# -*- coding: utf-8 -*-

import os
import sys

import pandas as pd

raw_phenotype: str = sys.argv[1]
processed_dir: str = sys.argv[2]

pt_samples: pd.DataFrame = pd.read_csv(
    raw_phenotype, sep = "\t"
).sort_values(by = "IID")

for col in pt_samples.columns[1:]:
    output_dir = os.path.join(processed_dir, col)
    phenotype_output: str = os.path.join(
        output_dir, "lipids_phenotype_" + col + ".csv"
    )
    genotype_id_output: str = os.path.join(
        output_dir, "id_list_" + col + ".csv"
    )

    phenotypes: pd.DataFrame = pt_samples[["IID", col]].dropna()
    phenotypes.to_csv(phenotype_output, index = False, header = True)
    phenotypes.IID.to_csv(genotype_id_output, index = False, header = True)

