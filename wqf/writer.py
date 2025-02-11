#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This module provides the target dataset writer.
"""

import uuid
from pathlib import Path
from typing import Any
from typing import Literal

import numpy as np
from dask.array import Array
from typing_extensions import override
from xarray import DataArray
from xarray import Dataset

from .interface.writing import Writing
from .progress import Progress

_KEY_CHUNKS: str = "config.wqf.writer.chunks"
"""
The key to configure chunking of data. The default is an empty
dictionary.
"""

_KEY_ENGINE: str = "config.wqf.writer.engine"
"""
The key to configure the writer engine. Possible engines are
`h5netcdf`, `netcdf4`, and `zarr`. The default is `zarr`.
"""

_KEY_ZLIB: str = "config.wqf.writer.zlib"
"""
The key to configure whether the data will be compressed using zlib.
The default is `true`.
"""

_KEY_COMPLEVEL: str = "config.wqf.writer.complevel"
"""
The key to configure the compression level. The default is `1`.
"""

_KEY_SHUFFLE: str = "config.wqf.writer.shuffle"
"""
The key to configure shuffling. The default is `true`.
"""


class Writer(Writing):
    """! The target dataset writer."""

    _config: dict[str, Any]
    """! The writer configuration."""

    _progress: bool
    """Displays a progress bar on the console, if set."""

    def __init__(
        self,
        config: dict[str, Any],
        chunks: dict[str:int] = None,
        engine: Literal["h5netcdf", "netcdf4", "zarr"] | None = None,
        progress: bool = False,
    ):
        """
        Creates a new writer instance.

        :param config: The writer configuration.
        :param chunks: An explicit configuration of chunks. Overrides the
        writer configuration.
        :param engine: An explicit specification of the writer engine.
        Overrides the writer configuration.
        :param progress: Displays a progress bar on the console, if set.
        """
        self._config = {
            _KEY_CHUNKS: {},
            _KEY_ENGINE: "h5netcdf",
            _KEY_ZLIB: "true",
            _KEY_COMPLEVEL: 1,
            _KEY_SHUFFLE: "true",
        }
        self._config.update(config)
        if chunks is not None:
            self._config[_KEY_CHUNKS].update(chunks)
        if engine is not None:
            self._config[_KEY_ENGINE] = engine
        self._progress = progress

    @override
    def write(
        self, dataset: Dataset, data_id: str | Path, **kwargs
    ):  # noqa: D102
        to_zarr = self._auto_engine(data_id) == "zarr"
        variables, encoding = self._encode(dataset, to_zarr)
        attrs = self._config.get("attrs", {})
        attrs.update(dataset.attrs)
        attrs["uuid"] = self._uuid

        with Dataset(variables, attrs=attrs) as ds:
            if to_zarr:
                with Progress(self._progress):
                    ds.to_zarr(data_id, encoding=encoding)
            else:
                with Progress(self._progress):
                    # noinspection PyTypeChecker
                    ds.to_netcdf(
                        data_id,
                        encoding=encoding,
                        format=self._format,
                        engine=self._auto_engine(data_id),
                    )

    def _encode(self, dataset: Dataset, to_zarr: bool = True):
        """This method does not belong to public API."""
        names: dict[str, str] = {}

        for vid, config in self._config.items():
            if vid not in dataset:
                continue
            names[vid] = self._get_name(config)

        variables: dict[str, DataArray] = {}
        encodings: dict[str, dict[str, Any]] = {}

        for vid, config in self._config.items():
            if vid not in dataset:
                continue

            dtype = self._get_dtype(config)
            attrs = self._encode_attrs(self._get_attrs(config, names), dtype)
            array = dataset[vid].data

            name = self._get_name(config)
            dims = self._get_dims(config)
            chunks: list[int] = []
            if name not in dims:  # not a coordinate dimension
                for i, dim in enumerate(dims):
                    if dim in self._chunks:
                        chunk_size = self._chunks[dim]
                        assert isinstance(chunk_size, int), (
                            f"Invalid chunk size specified for "
                            f"dimension '{dim}'"
                        )
                        if chunk_size == -1:
                            chunk_size = array.shape[i]
                        if chunk_size == 0:
                            chunk_size = array.chunksize[i]
                        chunks.append(chunk_size)
                    else:
                        chunks.append(array.chunksize[i])
            encodings[name] = self._encode_compress(
                dtype, attrs, chunks, to_zarr
            )
            variables[name] = self._encode_variable(name, dims, attrs, array)
        return variables, encodings

    def _auto_engine(self, path: str | Path) -> str:
        """This method does not belong to public API."""
        if f"{path}".endswith(".zarr"):
            return "zarr"
        if f"{path}".endswith(".nc"):
            return self._engine if self._engine != "zarr" else "h5netcdf"
        return self._engine

    @property
    def _chunks(self) -> dict[str, int]:
        """This method does not belong to public API."""
        return self._config[_KEY_CHUNKS]

    @property
    def _engine(self) -> Literal["h5netcdf", "netcdf4", "zarr"]:
        """This method does not belong to public API."""
        return self._config[_KEY_ENGINE]

    @property
    def _format(self) -> Literal["NETCDF4"]:
        """This method does not belong to public API."""
        return "NETCDF4"

    @property
    def _zlib(self) -> bool:
        """This method does not belong to public API."""
        return self._config[_KEY_ZLIB] == "true"

    @property
    def _complevel(self) -> int:
        """This method does not belong to public API."""
        return self._config[_KEY_COMPLEVEL]

    @property
    def _shuffle(self) -> bool:
        """This method does not belong to public API."""
        return self._config[_KEY_SHUFFLE] == "true"

    @property
    def _uuid(self) -> str:
        """This method does not belong to public API."""
        return f"{uuid.uuid4()}"

    @staticmethod
    def _get_attrs(config, names: dict[str:str]) -> dict[str, Any]:
        """This method does not belong to public API."""
        attrs = config["attrs"].copy()
        if "coordinates" in attrs:  # resolve coordinate variable identifiers
            coordinates = []
            for vid in attrs["coordinates"].split(" "):
                coordinates.append(names[vid])
            attrs["coordinates"] = " ".join(coordinates)
        return attrs

    @staticmethod
    def _get_dims(config: dict[str, Any]) -> list[str]:
        """This method does not belong to public API."""
        return config["dims"]

    @staticmethod
    def _get_dtype(config: dict[str, Any]) -> np.dtype:
        """This method does not belong to public API."""
        return np.dtype(config["dtype"])

    @staticmethod
    def _get_name(config: dict[str, Any]) -> str:
        """This method does not belong to public API."""
        return config["name"]

    @staticmethod
    def _encode_attrs(
        attrs: dict[str, Any], dtype: np.dtype
    ) -> dict[str, Any]:
        """This method does not belong to public API."""
        Writer._encode_attr(attrs, dtype, "valid_min")
        Writer._encode_attr(attrs, dtype, "valid_max")
        Writer._encode_attr(attrs, dtype, "_FillValue")
        Writer._encode_attr(attrs, dtype, "flag_values")
        Writer._encode_attr(attrs, dtype, "flag_masks")
        Writer._encode_attr(attrs, dtype, "precision")
        scaled_dtype = np.dtype("double")
        Writer._encode_attr(attrs, scaled_dtype, "scale_factor")
        Writer._encode_attr(attrs, scaled_dtype, "add_offset")
        return attrs

    @staticmethod
    def _encode_attr(attrs: dict[str, Any], dtype: np.dtype, name: str):
        """This method does not belong to public API."""
        if name in attrs:
            attrs[name] = Writer._convert(attrs[name], dtype)

    @staticmethod
    def _convert(value, dtype: np.dtype) -> np.ndarray:
        """This method does not belong to public API."""
        return np.array(value).astype(dtype).squeeze()

    @staticmethod
    def _encode_variable(
        name: str, dims: list[str], attrs: dict[str, Any], array: Array
    ) -> DataArray:
        """This method does not belong to public API."""
        return DataArray(data=array, dims=dims, name=name, attrs=attrs)

    def _encode_compress(
        self,
        dtype: np.dtype,
        attrs: dict[str:Any],
        chunks: list[int],
        to_zarr: bool = True,
    ) -> dict[str, Any]:
        """This method does not belong to public API."""
        enc = {"dtype": dtype}
        if "_FillValue" in attrs:
            enc["_FillValue"] = attrs.pop("_FillValue")
        if "add_offset" in attrs:
            enc["add_offset"] = attrs.pop("add_offset")
        if "scale_factor" in attrs:
            enc["scale_factor"] = attrs.pop("scale_factor")
        if chunks:
            if to_zarr:
                enc["chunks"] = tuple(chunks)
            else:
                enc["chunksizes"] = tuple(chunks)
        if to_zarr:
            pass
        else:
            enc["zlib"] = self._zlib
            enc["complevel"] = self._complevel
            enc["shuffle"] = self._shuffle
        return enc
