# -*- coding: utf-8 -*-

import importlib
import logging
from typing import Any, Callable, List, NoReturn, Optional, Union

from tensorboardX import SummaryWriter


class WriterTensorboardX():
    """
    TensorBoardX implementation.
    """
    def __init__(
        self,
        writer_dir: str,
        logger: logging.Logger,
        enable: bool
    ) -> None:
        self.writer: Optional[SummaryWriter] = None
        if enable:
            log_path: str = writer_dir
            try:
                # mypy returns an error saying SummaryWriter not found for
                # tensorboardX. Not sure why when the manual import above works.
                self.writer = importlib.import_module(
                    "tensorboardX"
                ).SummaryWriter(log_path)

            except ModuleNotFoundError:
                message = (
                    "Warning: TensorboardX visualization is configured to use, "
                    + "but currently not installed on this machine. \nPlease "
                    + "install the package by 'pip install tensorboardx' "
                    + "command or turn off the option in the 'config.json'"
                    + "file."
                )
                logger.warning(message)

        self.step: int = 0
        self.mode: str = ''

        self.tensorboard_writer_ftns: List[str] = [
            "add_scalar",
            "add_scalars",
            "add_image",
            "add_audio",
            "add_text",
            "add_histogram",
            "add_pr_curve",
            "add_embedding"
        ]


    def set_step(self, step: int, mode: str = "train") -> None:
        self.mode = mode
        self.step = step


    def __getattr__(self, name: str) -> Callable:
        """
        If visualization is configured for usage:
            return add_data() methods of tensorboard with additional information
            (step, tag) added.

        Otherwise:
            return blank function handle that does nothing
        """
        if name in self.tensorboard_writer_ftns:
            add_data: Optional[Callable] = getattr(self.writer, name, None)

            # TODO: Tensorboard object types, add later.
            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    add_data(
                        "{}/{}".format(self.mode, tag), data, self.step, *args,
                        **kwargs
                    )

            return wrapper

        else:
            def blank():
                pass


            raise AttributeError(
                "Type object 'WriterTensorboardX' has no attribute \
                '{}'".format(name)
            )

            return blank

