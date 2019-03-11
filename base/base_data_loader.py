# -*- coding: utf-8 -*-

from typing import Callable, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for custom data loaders.
    Performs automatic training and validation split based on given proportion.
    """
    def __init__(
        self,
        dataset: Dataset,
        validation_split: float,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn: Callable = default_collate
    ) -> None:
        self.validation_split: float = validation_split
        self.shuffle: bool = shuffle
        self.batch_idx: int = 0
        self.n_samples: int = len(dataset)

        # Setting training set and validation set samplers. Both samplers are
        # `None` if validation split proportion is set to 0.
        self.sampler: Optional[SubsetRandomSampler]
        self.valid_sampler: Optional[SubsetRandomSampler]
        self.sampler, self.valid_sampler = self._split_sampler(
            self.validation_split
        )

        # Named arguments for the torch data loader.
        self.init_kwargs: dict = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "collate_fn": collate_fn,
            "num_workers": num_workers
        }

        # Load the training set data with the training sampler.
        super(BaseDataLoader, self).__init__(
            sampler = self.sampler, **self.init_kwargs
        )


    def _split_sampler(
        self,
        split: float
    ) -> Tuple[Optional[SubsetRandomSampler], Optional[SubsetRandomSampler]]:
        """
        Randomly splits data indices into training and validation sets and
        creates the corresponding sampler objects.

        Parameters
        ----------
        split : float
            A float value dictating the split proportion of the validation set.

        Return
        ------
        tuple
            A tuple of the training and validation `SubsetRandomSampler`
            objects, in that order. A tuple of two `None` if split value is 0.
        """
        if split == 0.0:
            return None, None

        # Array of all sample indices.
        idx_full: np.ndarray = np.arange(self.n_samples)
        np.random.seed(0)
        np.random.shuffle(idx_full)

        len_valid: int = int(self.n_samples * split)

        # Indices for training and validation sets.
        valid_idx: np.ndarray = idx_full[0:len_valid]
        train_idx: np.ndarray = np.delete(idx_full, np.arange(0, len_valid))

        # Training and validation set sampler objects.
        train_sampler: SubsetRandomSampler = SubsetRandomSampler(train_idx)
        valid_sampler: SubsetRandomSampler = SubsetRandomSampler(valid_idx)

        # Turn off the shuffle option which is mutually exclusive with the
        # sampler option in the torch dataloader.
        self.shuffle: bool = False
        self.n_samples: int = len(train_idx)

        return train_sampler, valid_sampler


    def split_validation(self) -> Optional[DataLoader]:
        """
        Loads the validation data if data is split.

        Returns
        -------
        DataLoader
            A torch dataloader with the validation SubsetRandomSampler and
            initial arguments.
        """
        if self.valid_sampler is None:
            return None

        else:
            return DataLoader(sampler = self.valid_sampler, **self.init_kwargs)

