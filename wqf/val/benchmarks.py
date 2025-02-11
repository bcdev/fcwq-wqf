#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This module provides several benchmark methods used for product validation,
which is described in the Product Validation Plan.
"""
from typing import Any

import dask.array as da
from dask.array import Array
from xarray import DataArray
from xarray import Dataset

from ..interface.benchmark import Benchmark
from ..interface.constants import DID_LAT
from ..interface.constants import DID_LON
from ..interface.constants import DID_TIM
from ..interface.constants import VID_CHL
from ..interface.constants import VID_TIM
from ..readerfactory import ReaderFactory


class BGC(Benchmark):
    """A biogeochemical model as a benchmark."""

    def __init__(self, args):
        """Creates a new benchmark instance."""
        self._reader = ReaderFactory.create_reader(args.aws)
        self._data_id = args.bgcm_id

    def predict(
        self, cube: Dataset, *, min_pixels: int = 0, h: int = 1, n: int = 5
    ) -> tuple[DataArray, DataArray]:
        """
        Performs a BGC model forecast.

        :param cube: The reference dataset.
        :param min_pixels: The minimum number of observations in a time step.
        :param h: The forecast horizon.
        :param n: The forecast history.
        :return: The observed values and corresponding forecast values.
        """
        test: Dataset = self._reader.read(self._data_id, depth_level=3.0)
        ref = cube[VID_CHL][n + h - 1 :]
        pre = test[VID_CHL][n + h - 1 :]
        return align(ref, pre, min_pixels=min_pixels)


class Naive(Benchmark):
    """The naive benchmark method."""

    def predict(
        self, chl: DataArray, *, min_pixels: int = 0, h: int = 1, n: int = 5
    ) -> tuple[DataArray, DataArray]:
        """
        Performs the naive method on historic data.

        :param chl: The historic data.
        :param min_pixels: The minimum number of valid pixels in a time step.
        :param h: The forecast horizon.
        :param n: The forecast history.
        :return: The observed values and corresponding forecast values.
        """
        ref: DataArray = chl[n + h - 1 :]
        pre: DataArray = chl[n - 1 : -h]
        return align(ref, pre, min_pixels)


class SNaive(Benchmark):
    """The seasonal naive (s-naive) benchmark method."""

    def predict(
        self,
        chl: DataArray,
        *,
        min_pixels: int = 0,
        h: int = 1,
        n: int = 5,
    ) -> tuple[DataArray, DataArray]:
        """
        Performs the seasonal-naive method on historic data.

        :param chl: The historic data.
        :param min_pixels: The minimum number of observations in a time step.
        :param h: The forecast horizon.
        :param n: The forecast history.
        :return: The observed values and corresponding forecast values.
        """

        def ker(x: Array, **kwargs) -> Array:
            """The kernel function"""
            y = x[..., n - 1]
            for i in range(1, n):
                y = da.where(da.isfinite(y), y, x[..., n - i - 1])
            return y

        ref: DataArray = chl[n + h - 1 :]
        pre: DataArray = (
            chl[: 1 - n - h].rolling({DID_TIM: n}, min_periods=1).reduce(ker)
        )
        return align(ref, pre, min_pixels)


class MA(Benchmark):
    """The moving average (MA) benchmark method."""

    def predict(
        self, chl: DataArray, *, min_pixels: int = 0, h: int = 1, n: int = 5
    ) -> tuple[DataArray, DataArray]:
        """
        Performs the moving average method on historic data.

        :param chl: The historic data.
        :param min_pixels: The minimum number of observations in a time step.
        :param h: The forecast horizon.
        :param n: The forecast history.
        :return: The observed values and corresponding forecast values.
        """
        ref: DataArray = chl[n + h - 1 :]
        pre: DataArray = (
            chl[: 1 - n - h].rolling({DID_TIM: n}, min_periods=n).mean()
        )

        return align(ref, pre, min_pixels)


class SES(Benchmark):
    """The simple exponential smoothing (SES) benchmark method."""

    def predict(
        self,
        chl: DataArray,
        *,
        min_pixels: int = 0,
        h: int = 1,
        n: int = 5,
        a: Any = 0.8,
    ) -> tuple[DataArray, DataArray]:
        """
        Performs the simple exponential smoothing method on historic data.

        :param chl: The historic data.
        :param min_pixels: The minimum number of observations in a time step.
        :param h: The forecast horizon.
        :param n: The forecast history.
        :param a: The smoothing parameter.
        :return: The observed values and corresponding forecast values.
        """

        def ker(x: Array, **kwargs) -> Array:
            """The kernel function."""
            y = a * x[..., n - 1]
            for i in range(1, n):
                y += a * (1.0 - a) ** i * x[..., n - i - 1]
            return y

        ref: DataArray = chl[n + h - 1 :]
        pre: DataArray = (
            chl[: 1 - n - h].rolling({DID_TIM: n}, min_periods=n).reduce(ker)
        )
        return align(ref, pre, min_pixels)


class XGB(Benchmark):
    """An XGB model as a benchmark."""

    def __init__(self, args):
        """Creates a new benchmark instance."""
        self._reader = ReaderFactory.create_reader(args.aws)
        self._data_id = args.xgbm_id

    def predict(
        self, cube: Dataset, *, min_pixels: int = 0, h: int = 1
    ) -> tuple[DataArray, DataArray]:
        """
        Performs an XGB model forecast.

        :param cube: The reference dataset.
        :param min_pixels: The minimum number of observations in a time step.
        :param h: The forcast horizon.
        :return: The observed values and corresponding forecast values.
        """
        test: Dataset = self.read(h)
        ref = cube[VID_CHL]
        pre = test[VID_CHL]
        return align_nodata(ref, pre, min_pixels=min_pixels)

    def read(self, h: int) -> Dataset:
        """
        Reads an XGB forecast model test dataset.

        :param h: The forecast horizon.
        :return: The dataset.
        """
        return self._reader.read(self._data_id.replace("xgbH", f"xgb{h}"))


def align(ref, pre, min_pixels: int = 0):
    """Returns mutually aligned reference and forecast data."""
    return align_nodata(ref, align_coords(ref, pre), min_pixels)


def align_coords(ref: DataArray, pre: DataArray) -> DataArray:
    """
    Returns a forecast data array which has the same coordinates as the
    reference.

    Alignment of coordinates is necessary to align the time coordinate
    labels for primitive benchmarks and for BGC model data, which have
    somewhat different latitude and longitude coordinate labels (for
    unknown reasons).
    """
    assert (
        ref.shape == pre.shape
    ), f"shapes do not match: {ref.shape} != {pre.shape}"

    return DataArray(
        data=(
            pre.data
            if pre.chunks == ref.chunks
            else pre.data.rechunk(ref.chunks)
        ),
        coords=ref.coords,
        dims=ref.dims,
        attrs=ref.attrs,
    )


def align_nodata(
    ref: DataArray,
    pre: DataArray,
    min_pixels: int = 0,
) -> tuple[DataArray, DataArray]:
    """
    Returns mutually nullified reference and forecast data.
    """
    assert (
        ref.shape[1:] == pre.shape[1:]
    ), f"shapes do not match: {ref.shape[1:]} != {pre.shape[1:]}"
    assert (
        ref.chunks[1:] == pre.chunks[1:]
    ), f"chunks do not match: {ref.chunks[1:]} != {pre.chunks[1:]}"

    ref_period = _period(ref)
    pre_period = _period(pre)
    if ref_period != pre_period:
        beg = max(ref_period[0], pre_period[0])
        end = min(ref_period[1], pre_period[1])
        ref = ref.sel({DID_TIM: slice(beg, end)})
        pre = pre.sel({DID_TIM: slice(beg, end)})
    if min_pixels > 0:
        msk = da.full(ref.shape, True, chunks=ref.chunks)
        msk[ref.count([DID_LAT, DID_LON]).compute() < min_pixels, :, :] = (
            False
        )
        return ref.where(msk), pre.where(msk)
    else:
        return ref, pre


def _period(a: DataArray):
    """Returns the time interval a data array refers to."""
    return a.coords[VID_TIM][0], a.coords[VID_TIM][-1]
