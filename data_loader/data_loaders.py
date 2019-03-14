# -*- coding: utf-8 -*-

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
        chunksize: int,
        validation_split: float,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        training: bool = True
    ) -> None:
        self.data_dir: str = data_dir
        self.dataset: Dataset = LipidDataset(
            root = self.data_dir, train = training, chunksize = chunksize
        )

        super(LipidDataLoader, self).__init__(
            dataset = self.dataset,
            validation_split = validation_split,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = num_workers
        )

