#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""This module provides the source dataset reader."""

from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import xarray as xr
from typing_extensions import override
from xarray import Dataset

from .interface.constants import DID_DEP
from .interface.constants import VID_CHL
from .interface.constants import VID_NO3
from .interface.reading import Reading

_KEY_CHUNKS: str = "config.wqf.reader.chunks"
"""
The key to configure chunking. The default is `{}`.
"""

_KEY_ENGINE: str = "config.wqf.reader.engine"
"""
The key to configure the reader engine. Possible engines are `h5netcdf`,
`netcdf4`, and `zarr`. The default is `zarr`.
"""

_KEY_DECODE_CF: str = "config.wqf.reader.decode_cf"
"""
The key to configure whether to decode variables, assuming they were
saved according to CF conventions. The default is `true`.
"""

_KEY_DECODE_COORDS: str = "config.wqf.reader.decode_coords"
"""
The key to configure which variables are set as coordinate variables.
If `true` variables referred to in the coordinates attribute of the
datasets or individual variables are considered coordinate variables.
The default is `true`."""

_KEY_DECODE_TIMES: str = "config.wqf.reader.decode_times"
"""
The key to configure whether to decode times encoded in the standard
netCDF datetime format into datetime objects. The default is `true`.
"""

_KEY_DECODE_TIMEDELTA: str = "config.wqf.reader.decode_timedelta"
"""
The key to configure whether to decode variables and coordinates with
time units in {"days", "hours", "minutes", "seconds", "milliseconds",
"microseconds"} into timedelta objects. The default is `false`.
"""

_KEY_USE_CFTIME: str = "config.wqf.reader.use_cftime"
"""
The key to configure whether to decode times into `cftime.datetime`
objects or into `np.datetime64[ns]` objects. The default is `true`.
"""

_KEY_CONCAT_CHARACTERS: str = "config.wqf.reader.concat_characters"
"""
The key to configure concatenation along the last dimension of
character arrays to form string arrays. The default is `true`.
"""

_KEY_INLINE_ARRAY: str = "config.wqf.reader.inline_array"
"""
The key to configure whether to inline arrays directly in the dask
task graph. The default is `false`.
"""


class Reader(Reading):
    """The source dataset reader."""

    _config: dict[Any, Any]
    """The reader configuration."""

    def __init__(self, config: dict[str:Any] | None = None):
        """
        Creates a new reader instance.

        :param config: The reader configuration.
        """
        self._config = {
            _KEY_CHUNKS: {},
            _KEY_ENGINE: "zarr",
            _KEY_DECODE_CF: "true",
            _KEY_DECODE_COORDS: "true",
            _KEY_DECODE_TIMES: "true",
            _KEY_DECODE_TIMEDELTA: "true",
            _KEY_USE_CFTIME: "false",
            _KEY_CONCAT_CHARACTERS: "true",
            _KEY_INLINE_ARRAY: "false",
        }
        if config is not None:
            self._config.update(config)

    @override
    def read(
        self,
        data_id: str | Path,
        *,
        depth_level: Any = None,
        unify: bool = True,
    ) -> Dataset:
        """
        Reads a dataset.

        :param data_id: The dataset identifier, e.g., a file path.
        :param depth_level: The depth level (m) considered.
        :param unify: Unify dimensions and chunks of all variables.
        :returns: The dataset read.
        """
        ds = self._open(data_id)
        if DID_DEP in ds.dims and depth_level is not None:
            ds = ds.sel({DID_DEP: depth_level}, drop=True)
        if unify:
            ds = ds.broadcast_like(ds[VID_CHL]).unify_chunks()
        ds = self._nullify(ds)
        return ds

    def _open(self, data_id: str | Path) -> Dataset:
        """This method does not belong to public API."""
        kwargs = {}
        return xr.open_dataset(
            data_id,
            chunks=self._chunks,
            engine=self._auto_engine(data_id),
            mask_and_scale=True,
            decode_cf=self._decode_cf,
            decode_coords=self._decode_coords,
            decode_times=self._decode_times,
            decode_timedelta=self._decode_timedelta,
            use_cftime=self._use_cftime,
            concat_characters=self._concat_characters,
            inline_array=self._inline_array,
            backend_kwargs=kwargs,
        ).astype(np.single, copy=False)

    @staticmethod
    def _nullify(ds: Dataset):
        """Nullifies data where applicable."""
        if VID_NO3 in ds and DID_DEP not in ds.dims:
            condition = da.isfinite(ds[VID_NO3].data)
            for name, variable in ds.data_vars.items():
                if name != VID_CHL:  # chlorophyll only
                    continue
                variable.data = da.where(condition, variable.data, np.nan)
        return ds

    def _auto_engine(self, data_id: str | Path) -> str:
        """This method does not belong to public API."""
        if f"{data_id}".endswith(".zarr"):
            return "zarr"
        if f"{data_id}".endswith(".nc"):
            return self._engine if self._engine != "zarr" else "h5netcdf"
        return self._engine

    @property
    def _chunks(self) -> dict[str:int]:
        """This method does not belong to public API."""
        return self._config[_KEY_CHUNKS]

    @property
    def _engine(self) -> str:
        """This method does not belong to public API."""
        return self._config[_KEY_ENGINE]

    @property
    def _decode_cf(self) -> bool:
        """This method does not belong to public API."""
        return self._config[_KEY_DECODE_CF] == "true"

    @property
    def _decode_coords(self) -> bool:
        """This method does not belong to public API."""
        return self._config[_KEY_DECODE_COORDS] == "true"

    @property
    def _decode_times(self) -> bool:
        """This method does not belong to public API."""
        return self._config[_KEY_DECODE_TIMES] == "true"

    @property
    def _decode_timedelta(self) -> bool:
        """This method does not belong to public API."""
        return self._config[_KEY_DECODE_TIMEDELTA] == "true"

    @property
    def _use_cftime(self) -> bool:
        """This method does not belong to public API."""
        return self._config[_KEY_USE_CFTIME] == "true"

    @property
    def _concat_characters(self) -> bool:
        """This method does not belong to public API."""
        return self._config[_KEY_CONCAT_CHARACTERS] == "true"

    @property
    def _inline_array(self) -> bool:
        """This method does not belong to public API."""
        return self._config[_KEY_INLINE_ARRAY] == "true"
