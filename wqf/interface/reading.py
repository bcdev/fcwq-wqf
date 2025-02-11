#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""! This module provides the reading interface."""

from abc import ABC
from abc import abstractmethod
from pathlib import Path

from xarray import Dataset


class Reading(ABC):
    """! The dataset reading interface."""

    @abstractmethod
    def read(self, data_id: str | Path, **kwargs) -> Dataset:
        """
        Reads a dataset.

        :param data_id: The dataset identifier, e.g., a file path.
        :param kwargs: Any optional keyword arguments.
        :returns: The dataset read.
        """
