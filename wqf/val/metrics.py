#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This module provides several metrics used for product validation, which is
described in the Product Validation Plan.
"""
from numbers import Number

import dask.array as da
import numpy as np
from xarray import DataArray

from ..interface.constants import DID_LAT
from ..interface.constants import DID_LON
from ..interface.constants import DID_TIM
from ..interface.metric import Metric


def _select(
    ref: DataArray, pre: DataArray, condition: DataArray | None
) -> tuple[DataArray, DataArray]:
    """Returns the data which satisfy the given condition."""
    if condition is not None:
        pre = pre.where(condition)
        ref = ref.where(condition)
    return ref, pre


class Bias(Metric):
    """
    The bias.

    The mean error is termed bias. The closer the bias is to zero,
    the better is the forecast model.
    """

    def value(self, ref: DataArray, pre: DataArray, **kwargs) -> Number:
        return Bias.err(ref, pre).mean().values.item()

    def image(self, ref: DataArray, pre: DataArray, **kwargs) -> DataArray:
        return Bias.err(ref, pre).mean(DID_TIM)

    def series(self, ref: DataArray, pre: DataArray, **kwargs) -> DataArray:
        return Bias.err(ref, pre).mean([DID_LAT, DID_LON])

    @staticmethod
    def err(
        ref: DataArray, pre: DataArray, condition: DataArray | None = None
    ) -> DataArray:
        """Returns the error."""
        ref, pre = _select(ref, pre, condition)
        return -(ref - pre.data)  # sign is important

    @staticmethod
    def rer(
        ref: DataArray, pre: DataArray, condition: DataArray | None = None
    ) -> DataArray:
        """Returns the relative error."""
        ref, pre = _select(ref, pre, condition)
        return Bias.err(pre, ref) / ref


class Count(Metric):
    """
    The number counts.
    """

    def value(self, ref: DataArray, pre: DataArray, *kwargs) -> Number:
        return da.isfinite(ref - pre.data).sum().values.item()

    def image(self, ref: DataArray, pre: DataArray, **kwargs) -> DataArray:
        counts = (
            da.isfinite(ref - pre.data)
            .sum(DID_TIM)
            .astype(np.single, copy=False)
        )
        return counts.where(counts > 0.0)

    def series(self, ref: DataArray, pre: DataArray, **kwargs) -> DataArray:
        counts = (
            da.isfinite(ref - pre.data)
            .sum([DID_LAT, DID_LON])
            .astype(np.single, copy=False)
        )
        return counts.where(counts > 0.0)


class MAD(Metric):
    """
    The median absolute deviation (MAD).
    """

    def value(self, ref: DataArray, pre: DataArray, **kwargs) -> Number:
        return MAD.ad(ref, pre).compute().median().values.item()

    def image(self, ref: DataArray, pre: DataArray, **kwargs) -> DataArray:
        return MAD.ad(ref, pre).median(DID_TIM)

    def series(self, ref: DataArray, pre: DataArray, **kwargs) -> DataArray:
        return MAD.ad(ref, pre).compute().median([DID_LAT, DID_LON])

    @staticmethod
    def ad(
        ref: DataArray, pre: DataArray, condition: DataArray | None = None
    ) -> DataArray:
        """Returns the absolute deviation."""
        ref, pre = _select(ref, pre, condition)
        return da.abs(ref - pre.data)


class MAPD(Metric):
    """
    The median absolute percentage deviation (MAPD).

    This metric actually computes the median absolute relative deviation.
    Multiply the returned metrics with 100 to obtain percentage. The closer
    the MAPD is to zero, the better is the forecast model.
    """

    def value(
        self,
        ref: DataArray,
        pre: DataArray,
        *,
        condition: DataArray | None = None,
    ) -> Number:
        return MAPD.apd(ref, pre, condition).compute().median().values.item()

    def image(
        self,
        ref: DataArray,
        pre: DataArray,
        *,
        condition: DataArray | None = None,
    ) -> DataArray:
        return MAPD.apd(ref, pre, condition).median(DID_TIM)

    def series(
        self,
        ref: DataArray,
        pre: DataArray,
        *,
        condition: DataArray | None = None,
    ) -> DataArray:
        return (
            MAPD.apd(ref, pre, condition).compute().median([DID_LAT, DID_LON])
        )

    @staticmethod
    def apd(
        ref: DataArray, pre: DataArray, condition: DataArray | None = None
    ) -> DataArray:
        """Returns the absolute percentage deviation."""
        ref, pre = _select(ref, pre, condition)
        return da.abs((ref - pre.data) / ref)


class R2(Metric):
    """
    The coefficient of determination (R²).

    The closer the coefficient of determination is to unity, the
    better is the forecast model.

    Wikipedia contributors. "Coefficient of determination." Wikipedia,
    The Free Encyclopedia. Wikipedia, The Free Encyclopedia, 17 Nov. 2024.
    Web. 17 Nov. 2024. https://w.wiki/3kMS.
    """

    def value(self, ref: DataArray, pre: DataArray, **kwargs) -> Number:
        ref, pre = _select(ref, pre, da.isfinite(ref - pre.data))
        ssr = da.square(ref - pre.data).sum()
        sst = da.square(ref - ref.mean()).sum()
        return (1.0 - ssr / sst).values.item()

    def image(self, ref: DataArray, pre: DataArray, **kwargs) -> DataArray:
        ref, pre = _select(ref, pre, da.isfinite(ref - pre.data))
        ssr = da.square(ref - pre.data).sum(DID_TIM)
        sst = da.square(ref - ref.mean(DID_TIM)).sum(DID_TIM)
        ssr, sst = _select(ssr, sst, ssr > 0.0)
        return 1.0 - ssr / sst

    def series(self, ref: DataArray, pre: DataArray, **kwargs) -> DataArray:
        ref, pre = _select(ref, pre, da.isfinite(ref - pre.data))
        ssr = da.square(ref - pre.data).sum([DID_LAT, DID_LON])
        sst = da.square(ref - ref.mean([DID_LAT, DID_LON])).sum(
            [DID_LAT, DID_LON]
        )
        return 1.0 - ssr / sst


class RMSE(Metric):
    """
    The root mean squared error (RMSE).

    The closer the RMSE is to zero, the better is the forecast model.
    """

    def value(self, ref: DataArray, pre: DataArray, **kwargs) -> Number:
        return da.sqrt(RMSE.se(ref, pre).mean()).values.item()

    def image(self, ref: DataArray, pre: DataArray, **kwargs) -> DataArray:
        return da.sqrt(RMSE.se(ref, pre).mean(DID_TIM))

    def series(self, ref: DataArray, pre: DataArray, **kwargs) -> DataArray:
        return da.sqrt(RMSE.se(ref, pre).mean([DID_LAT, DID_LON]))

    @staticmethod
    def se(ref: DataArray, pre: DataArray) -> DataArray:
        """Returns the squared error."""
        return da.square(ref - pre.data)


class WRMSSE(Metric):
    """
    The weighted root mean squared scaled error (WRMSSE).

    The metric computes a weighted root mean squared scaled error
    with equal weights for each time series. The closer the value
    is to zero, the better is the forecast model. Good models achieve
    a WRMSSE less than unity.
    """

    def value(
        self,
        ref: DataArray,
        pre: DataArray,
        *,
        condition: DataArray | None = None,
        b: int = 5,
        h: int = 1,
    ) -> Number:
        return WRMSSE.rmsse(ref, pre, condition, b, h).mean().values.item()

    def image(
        self,
        ref: DataArray,
        pre: DataArray,
        *,
        condition: DataArray | None = None,
        b: int = 5,
        h: int = 1,
    ) -> DataArray:
        return WRMSSE.rmsse(ref, pre, condition, b, h).mean(DID_TIM)

    def series(
        self,
        ref: DataArray,
        pre: DataArray,
        *,
        condition: DataArray | None = None,
        b: int = 5,
        h: int = 1,
    ) -> DataArray:
        return WRMSSE.rmsse(ref, pre, condition, b, h).mean(
            [DID_LAT, DID_LON]
        )

    @staticmethod
    def rmsse(
        ref: DataArray,
        pre: DataArray,
        condition: DataArray | None,
        b: int,
        h: int,
    ) -> DataArray:
        """Returns the root mean squared scaled error."""
        ref, pre = _select(ref, pre, condition)
        fwd = WRMSSE._fwd_mean_squared_diff(h, ref, pre)
        bwd = WRMSSE._bwd_mean_squared_diff(b, ref)
        return da.sqrt(
            fwd[b + 1 + h // 2 : fwd.shape[0] - h // 2]
            / bwd[b // 2 : bwd.shape[0] - b // 2 - h]
        )

    @staticmethod
    def _bwd_mean_squared_diff(b: int, ref: DataArray) -> DataArray:
        return (
            da.square(ref.diff(DID_TIM))
            .rolling({DID_TIM: b}, min_periods=b)
            .mean()
            .drop_vars(DID_TIM)
        )

    @staticmethod
    def _fwd_mean_squared_diff(
        h: int, ref: DataArray, pre: DataArray
    ) -> DataArray:
        return (
            da.square(ref - pre.data)
            .rolling({DID_TIM: h}, min_periods=h)
            .mean()
        )
