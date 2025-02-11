#  Copyright (c) Brockmann Consult, 2024
#  License: MIT

"""
This module provides the algorithm to forecast water quality.
"""

from pathlib import Path

import dask.array as da
import numpy as np
from typing_extensions import override
from xgboost import Booster
from xgboost import DMatrix

from .. import xgb
from ..interface.algorithm import BlockAlgorithm
from ..interface.constants import VID_CHL
from ..interface.constants import VID_NO3


class Forecast(BlockAlgorithm):
    """
    The algorithm to forecast water quality.

    The algorithm makes use of an extreme gradient-boosted
    decision tree.
    """

    _columns: list[tuple[int, tuple[int, str]]]
    """
    The list of forecast model input data columns.

    Each column is specified by `tuple[key: int, value: tuple[int, str]]`.
    The key is the column index and the value is a tuple of historic day
    enumerator `[0, -1, -2, ...]` and feature name.
    """

    _h: int
    """The forecast horizon (days)."""

    _model_spec: str | Path
    """The path to the forecast model file."""

    _model: Booster
    """The forecast model."""

    _nthread: int
    """The number of threads used by the forecast model."""

    _test: bool
    """Operate in test mode for verification and validation."""

    def __init__(
        self,
        model_spec: str | Path,
        horizon: int = 1,
        nthread: int = 1,
        test: bool = False,
    ):
        """
        Creates a new algorithm.

        :param model_spec: The forecast model specifier.
        :param horizon: The forecast horizon (days).
        :param nthread: The number of threads used by the forecast model.
        :param test: Operate in test mode.
        """
        super().__init__(np.dtype("single"))
        self._model_spec = model_spec
        self._model, self._columns = Forecast.load_model(model_spec)
        self._h = horizon
        self._nthread = nthread
        self._test = test

    def __reduce__(self):
        """
        For serialization.

        Does not serialize the forecast model by intention.
        """
        return self.__class__, (
            self._model_spec,
            self._h,
            self._nthread,
            self._test,
        )

    @override
    def chunks(self, *inputs: da.Array) -> tuple[int, ...] | None:
        n = (
            inputs[0].chunksize[0] - self.history - self.horizon + 1
            if self._test
            else self.horizon
        )
        return (n,) + inputs[0].chunks[1:]

    @property
    @override
    def created_axes(self) -> list[int] | None:
        return [0]

    @property
    @override
    def dropped_axes(self) -> list[int]:
        return [0]

    def forecast(self, *feats: np.ndarray, names: list[str]) -> np.ndarray:
        """
        Computes the water quality forecast.

        :param feats: The input features.
        :param names: The input feature names.
        :return: The water quality forecast.
        """
        if not self._test:
            f: dict[str, np.ndarray] = {
                name: feat.astype(
                    self.dtype, copy=name == VID_CHL  # copy for overwrite
                )
                for name, feat in zip(names, feats)
            }
            return self.fc(f)

        h = self.horizon
        n = self.history
        y = np.empty(
            (feats[0].shape[0] - n - h + 1,) + feats[0].shape[1:], self.dtype
        )
        for t in range(y.shape[0]):
            f: dict[str, np.ndarray] = {
                name: feat[t : t + n + h].astype(
                    self.dtype, copy=name == VID_CHL  # copy for overwrite
                )
                for name, feat in zip(names, feats)
            }
            y[t, :, :] = self.fc(f)[-1, :, :]
        return y

    def fc(self, f: dict[str, np.ndarray]) -> np.ndarray:
        """
        Computes the water quality forecast for multiple time steps into
        the future. Implements ATBD eq. (2-2).

        :param f: The feature dictionary.
        :return: The water quality forecast.
        """
        H = self.horizon  # noqa
        y = f[VID_CHL]
        z = f[VID_NO3]
        for h in range(H, 0, -1):
            x = self.table(f, h)
            y[-h, :, :] = self._predict(x).reshape(y.shape[1:])
        y[~np.isfinite(z)] = self.nan

        return y[-H:, :, :]

    def _predict(self, x: DMatrix) -> np.ndarray:
        """
        Returns the predicted values.

        :param x: The predictor values.
        :return: The predicted values (negative values are clipped).
        """
        return np.maximum(
            0.0, self._model.predict(x, validate_features=False)
        )

    compute_block = forecast

    @property
    def history(self) -> int:
        """
        Returns the absolute number of historic time steps needed
        for a forecast.
        """
        n = 0
        for _, (t, __) in self._columns:
            if t < n:
                n = t
        return -n

    @property
    def horizon(self) -> int:
        """Returns the forecast horizon (days)."""
        return self._h

    @property
    @override
    def name(self) -> str:
        return "forecast"

    @staticmethod
    def load_model(
        spec: str | Path,
    ) -> tuple[Booster, list[tuple[int, tuple[int, str]]]]:
        """This method does not belong to public API."""
        reg = xgb.registry()
        booster = Booster(model_file=reg.file(spec) if spec in reg else spec)
        columns = {}
        for i, name in enumerate(booster.feature_names):
            if name.startswith("t-"):
                columns[i] = intstr(
                    *name.removeprefix("t").split("_", maxsplit=1)
                )
            else:
                columns[i] = 0, name
        return booster, sorted(columns.items())

    def table(self, f: dict[str : np.ndarray], h: int) -> DMatrix:
        """This method does not belong to public API."""
        x = np.stack(
            [np.ravel(f[name][t - h, :, :]) for _, (t, name) in self._columns]
        )
        return DMatrix(x.T, nthread=self._nthread, silent=True)


def intstr(t: str, name: str) -> tuple[int, str]:
    """This method does not belong to public API."""
    return int(t), name
