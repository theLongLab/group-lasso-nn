# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
from torchvision import datasets, transforms

from base import BaseDataLoader


class LipidGenotypeDataLoader(BaseDataLoader):
    """
    """
    def __init(
        self,
        data_dir: str,
        validation_split: float,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        training: bool = True
    ) -> None:
        transformation = None

        self.data_dir: str = data_dir
        self.dataset: Dataset = None

        super(LipidGenotypeDataLoader, self).__init__(
           dataset = self.dataset,
           validation_split = validation_split,
           batch_size = batch_size,
           shuffle = shuffle,
           num_workers = num_workers
       )


# class ExampleDataLoader(BaseDataLoader):
#     """
#     Example data loader with MNIST data.
#     Comment out the class and build custom data loader when using for own
#     purposes.
#     """
#     def __init__(
#         self,
#         data_dir: str,
#         validation_split: float,
#         batch_size: int,
#         shuffle: bool,
#         num_workers: int,
#         training: bool = True
#     ) -> None:
#         # Normalization values taken from the PyTorch MNIST example.
#         transformation: transforms.Compose = transforms.Compose([
#             transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
#         ])

#         self.data_dir: str = data_dir
#         self.dataset: Dataset = datasets.MNIST(
#             root = self.data_dir,
#             train = training,
#             download = True,
#             transform = transformation
#         )

#         super(ExampleDataLoader, self).__init__(
#             dataset = self.dataset,
#             validation_split = validation_split,
#             batch_size = batch_size,
#             shuffle = shuffle,
#             num_workers = num_workers
#         )

