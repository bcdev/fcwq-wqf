#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""This module defines the processing interface."""

from abc import ABC
from abc import abstractmethod
from argparse import Namespace
from typing import Any

from xarray import Dataset


class Processing(ABC):
    """The processing interface."""

    @abstractmethod
    def get_config_package(self) -> str:
        """Returns the name of the processor's configuration package."""

    @abstractmethod
    def get_default_config(self) -> dict[str:Any]:
        """Returns the processor's default configuration."""

    @abstractmethod
    def get_name(self) -> str:
        """Returns the name of the processor."""

    @abstractmethod
    def get_version(self) -> str:
        """Returns the version of the processor."""

    @abstractmethod
    def run(self, args: Namespace) -> None:
        """
        Runs the processor with supplied arguments.

        :param args: The processing arguments.
        """

    @abstractmethod
    def get_result(self, args: Namespace, *inputs: Dataset) -> Dataset:
        """
        Returns the result dataset.

        :param args: The processing arguments.
        :param inputs: The processing input datasets.
        :return: The processing result dataset.
        """
