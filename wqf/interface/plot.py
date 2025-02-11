#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This module defines plot interface.
"""
from abc import ABCMeta
from abc import abstractmethod
from typing import Any

from matplotlib.figure import Figure
from xarray import DataArray


class Plot(metaclass=ABCMeta):
    """The plot interface."""

    @abstractmethod
    def plot(
        self,
        data: DataArray | tuple[DataArray, DataArray],
        xlabel: str | None = None,
        ylabel: str | None = None,
        xlim: tuple[Any, Any] | None = None,
        ylim: tuple[Any, Any] | None = None,
        title: str | None = None,
        fn: str | None = None,
        show: bool = False,
        **kwargs,
    ) -> Figure:
        """
        Plots the given data.

        :param data: The data to plot.
        :param xlabel: The label for the x-axis.
        :param ylabel: The label for the y-axis.
        :param xlim: The limits for the x-axis.
        :param ylim: The limits for the y-axis.
        :param title: The title of the plot.
        :param fn: The file name to save the figure.
        :param show: Show the figure.
        :param kwargs: Additional keyword arguments.
        :return: The plot.
        """
