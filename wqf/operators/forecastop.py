#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This module provides the forecast operator to produce a water quality
forecast product.
"""

from argparse import Namespace

import dask.array as da
from typing_extensions import override
from xarray import Dataset

from ..algorithms.forecast import Forecast
from ..algorithms.gaussian import Gaussian
from ..datasetbuilder import DatasetBuilder
from ..interface.constants import DID_LAT
from ..interface.constants import DID_LON
from ..interface.constants import DID_TIM
from ..interface.constants import VID_CHL
from ..interface.constants import VID_LAT
from ..interface.constants import VID_LON
from ..interface.constants import VID_TIM
from ..interface.operator import Operator
from ..logger import get_logger


class ForecastOp(Operator):
    """The forecast operator."""

    _args: Namespace
    """The configuration parameters."""

    _builder: DatasetBuilder
    """The builder to build the target dataset."""

    def __init__(self, args: Namespace):
        """
        Creates a new forecast operator instance.

        :param args: The configuration parameters.
        """
        self._args = args

    @property
    @override
    def name(self) -> str:  # noqa: D102
        return "forecast"

    @override
    def run(self, source: Dataset) -> Dataset:  # noqa: D102
        """
        Runs the operator.

        :param source: The source dataset.
        :return: The result dataset.
        """
        get_logger().debug(f"loading forecast model: {self._args.model}")
        f = Forecast(
            self._args.model,
            horizon=self._args.horizon,
            nthread=self._args.nthread,
            test=self._args.test,
        )
        feats, names = self._features(source)
        array = f.apply_to(*feats, names=names)
        if self._args.gaussian_filter is not None:
            g = Gaussian(array.dtype)
            array = g.apply_to(array, fwhm=self._args.gaussian_filter)

        builder = DatasetBuilder()
        for name, value in sorted(
            vars(self._args).items(), key=lambda item: item[0]
        ):
            builder.add_attr(name, f"{value}")

        builder.add_dim(DID_TIM, array.shape[0])
        builder.add_dim(DID_LAT, array.shape[1])
        builder.add_dim(DID_LON, array.shape[2])

        builder.add_var(VID_TIM, DID_TIM)
        builder.add_var(VID_LAT, DID_LAT)
        builder.add_var(VID_LON, DID_LON)
        builder.add_var(VID_CHL, (DID_TIM, DID_LAT, DID_LON))

        builder.add_array(VID_TIM, source[VID_TIM].data[-array.shape[0] :])
        builder.add_array(VID_LAT, source[VID_LAT].data)
        builder.add_array(VID_LON, source[VID_LON].data)
        builder.add_array(VID_CHL, array)

        return builder.build()

    @staticmethod
    def _features(source: Dataset) -> tuple[list[da.Array], list[str]]:
        """This method does not belong to public API"""
        feats, names = [], []
        for name, feat in source.data_vars.items():
            feats.append(feat.data)
            names.append(name)
        return feats, names
