#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This module defines the operator interface.

Operators employ algorithms to transform source datasets
into a single target dataset. Operators constitute the logical
building blocks of the WQF processor."""

from abc import ABC
from abc import abstractmethod

from xarray import Dataset


class Operator(ABC):
    """The operator interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the operator.

        :return: The name of the operator.
        """

    @abstractmethod
    def run(self, *inputs: Dataset, **kwargs) -> Dataset:
        """
        Runs the operator.

        :param inputs: The input datasets.
        :param kwargs: Any keyword arguments.
        :return: The result dataset.
        """
