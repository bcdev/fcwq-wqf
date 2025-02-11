#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This module defines the benchmark interface.
"""

from abc import ABCMeta
from abc import abstractmethod

from xarray import DataArray
from xarray import Dataset


class Benchmark(metaclass=ABCMeta):
    """
    The benchmark interface.

    A benchmark method performs time series forecasts using historic data
    of the predictor for evaluation by a performance metric.
    """

    @abstractmethod
    def predict(
        self, cube_like: Dataset | DataArray, **kwargs
    ) -> tuple[DataArray, DataArray]:
        """
        Performs time series forecasts on historic data.

        The results returned are for evaluation by a performance
        metric.

        :param cube_like: The historic data.
        :param kwargs: Any keyword arguments.
        :return: A tuple containing the observed values of the predictor
        and corresponding forecast values.
        """
