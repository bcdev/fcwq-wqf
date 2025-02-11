#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This module provides a reader/writer, which reads from and writes to
AWS team store.
"""

from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
from xarray import Dataset

from .interface.constants import DID_DEP
from .interface.constants import VID_CHL
from .interface.constants import VID_NO3
from .interface.reading import Reading
from .interface.writing import Writing


class AWS(Reading, Writing):
    """The AWS team store reader and writer."""

    _store: Any
    """The AWS team store."""

    _depth_level: Any
    """
    The selected depth level (m).
    """

    def __init__(
        self,
        bucket: str,
        key: str,
        secret: str,
    ):
        """
        Creates a new AWS team store reader/writer instance.

        :param bucket: The AWS store bucket name.
        :param key: The AWS store key.
        :param secret: The AWS store secret.
        """
        from xcube.core.store import DataStore
        from xcube.core.store import MutableDataStore
        from xcube.core.store import new_data_store

        self._store: DataStore | MutableDataStore = new_data_store(
            "s3",
            root=bucket,
            storage_options={
                "anon": False,
                "key": key,
                "secret": secret,
            },
        )

    def read(
        self,
        data_id: str | Path,
        *,
        depth_level: Any = None,
        unify: bool = True,
    ) -> Dataset:
        """
        Reads a dataset from AWS team store.

        :param data_id: The dataset identifier.
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

    def _open(self, data_id):
        """This method does not belong to public API."""
        return self._store.open_data(data_id).astype(np.single, copy=False)

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

    def write(
        self, dataset: Dataset, data_id: str | Path, *, replace: bool = False
    ):
        """
        Writes a dataset to AWS team store.

        :param dataset: The dataset to be written.
        :param data_id: The dataset identifier.
        :param replace: Whether to replace an existing dataset.
        """
        self._store.write_data(data=dataset, data_id=data_id, replace=replace)
