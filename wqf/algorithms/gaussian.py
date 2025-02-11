#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This module provides the algorithm to apply a Gaussian filter.
"""

from numbers import Number

from dask import array as da
from dask_image.ndfilters import gaussian
from typing_extensions import override

from ..interface.algorithm import Algorithm


class Gaussian(Algorithm):
    """
    Applies a lateral Gaussian filter to a data cube.

    The implementation does not propagate NaN.
    """

    @override
    def apply_to(self, cube: da.Array, *, sigma: Number = 1.0) -> da.Array:
        """
        Applies a lateral Gaussian filter to a data cube.

        :param cube: The data cube.
        :param sigma: The standard deviation of the Gaussian kernel (pixels).
        :return: The filtered data cube.
        """
        m = da.isnan(cube)
        v = da.where(m, 0.0, cube)
        w = da.where(m, 0.0, 1.0)
        v = gaussian(v, [0.0, sigma, sigma], mode="constant")
        w = gaussian(w, [0.0, sigma, sigma], mode="constant")
        return da.where(m, cube, v / w)

    @property
    @override
    def name(self) -> str:
        return "gaussian"
