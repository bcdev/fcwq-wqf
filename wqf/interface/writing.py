#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""This module provides the dataset writing interface."""

from abc import ABC
from abc import abstractmethod
from pathlib import Path

from xarray import Dataset


class Writing(ABC):
    """! The dataset writing interface."""

    @abstractmethod
    def write(self, dataset: Dataset, data_id: str | Path, **kwargs):
        """
        Writes a dataset.

        :param dataset: The dataset to be written.
        :param data_id: The dataset identifier.
        :param kwargs: Any optional keyword arguments.
        """
