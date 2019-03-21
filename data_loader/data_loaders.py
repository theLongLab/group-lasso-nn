# -*- coding: utf-8 -*-

from typing import Optional

from torch.utils.data import Dataset
from torchvision import datasets, transforms

from base import BaseDataLoader
from datasets import LipidDataset


class LipidDataLoader(BaseDataLoader):
    """
    Lipid data loader.
    """
    def __init__(
        self,
        data_dir: str,
        validation_split: float,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        block_size: Optional[float] = None,
        dask_sample: Optional[float] = None,
        training: bool = True
    ) -> None:
        self.data_dir: str = data_dir
        if dask_sample is not None:
            dask_sample = int(dask_sample)

        self.dataset: Dataset = LipidDataset(
            root = self.data_dir,
            train = training,
            block_size = block_size,
            dask_sample = dask_sample
        )

        super(LipidDataLoader, self).__init__(
            dataset = self.dataset,
            validation_split = validation_split,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = num_workers
        )

