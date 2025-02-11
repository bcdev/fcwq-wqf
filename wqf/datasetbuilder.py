#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This module provides a class to build datasets based on the
Common Data Language (CDL) model.
"""

from typing import Any

import dask.array as da
from dask.array import Array
from xarray import DataArray
from xarray import Dataset


class DatasetBuilder:
    """A class to build datasets."""

    _attrs_dict: dict[str, Any]
    """The dictionary of global attributes."""
    _array_dict: dict[str, Array]
    """The dictionary of Dask arrays included with the dataset."""
    _dimid_dict: dict[str, Any]
    """The dictionary of Dask array dimension identifiers."""
    _shape_dict: dict[str, int]
    """The dictionary of Dask array dimension sizes."""
    _chunk_dict: dict[str, int]
    """The dictionary of Dask array chunk sizes."""

    def __init__(self):
        """Creates a new dataset builder instance."""
        self._attrs_dict = {}
        self._array_dict = {}
        self._dimid_dict = {}
        self._shape_dict = {}
        self._chunk_dict = {}

    def add_attr(self, name: str, value: Any):
        """
        Adds a global attribute to the dataset.

        :param name: The attribute name.
        :param value: The attribute value.
        :return: The builder itself.
        """
        self._attrs_dict[name] = value
        return self

    def add_array(self, vid: str, array: Array):
        """
        Adds an array to a variable.

        :param vid: The variable identifier.
        :param array: The array.
        :return: The builder itself.
        """
        assert (
            vid not in self._array_dict
        ), f"Variable ID '{vid}' is already associated with an array"
        return self._add_array(vid, array)

    def _add_array(self, vid: str, array: Array):
        """This method does not belong to public API."""
        assert (
            vid in self._dimid_dict
        ), f"Variable ID '{vid}' is not associated with dimensions"
        for i, d in enumerate(self._dimid_dict[vid]):
            shape = array.shape[i]
            if d in self._shape_dict:
                assert (
                    shape == self._shape_dict[d]
                ), f"Invalid array shape {shape} for variable {vid}"
            else:
                self._shape_dict[d] = shape
            if isinstance(array, Array):
                chunk = array.chunksize[i]
                if d not in self._chunk_dict or chunk < self._chunk_dict[d]:
                    self._chunk_dict[d] = chunk
        self._array_dict[vid] = array
        return self

    def add_dataset_array(self, vid: str, uid: str, dataset: Dataset):
        """!
        Adds an array from an existing dataset to a variable.

        :param vid: The identifier of the builder variable.
        :param uid: The name of the dataset variable the data of which
        will be associated with the variable.
        :param dataset: The dataset.
        """
        self.add_array(vid, dataset[uid].data)

    def get_array(self, vid: str) -> Array:
        """
        Returns the array associated with a given variable identifier.

        :param vid: The variable identifier.
        :return: The array.
        """
        assert (
            vid in self._array_dict
        ), f"Variable ID '{vid}' is not associated with an array"
        return self._array_dict[vid]

    def add_var(self, vid: str, did: str | tuple[str, ...] | None = None):
        """
        Adds a variable.

        :param vid: The variable identifier.
        :param did: The dimension identifiers.
        :return: The builder itself.
        """
        self._assert_vid_is_undefined(vid)
        if did is None:
            did = ()
        if isinstance(did, str):
            did = (did,)
        for d in did:
            self._assert_did_is_defined(d, vid)
        self._dimid_dict[vid] = did
        return self

    def add_dim(self, did: str, shape: int, chunk: int = -1):
        """
        Adds a dimension to the dataset.

        :param did: The dimension identifier.
        :param shape: The dimension shape.
        :param chunk: The dimension chunk size.
        :return: The builder itself.
        """
        self._assert_did_is_undefined(did)
        self._shape_dict[did] = shape
        if chunk == -1:
            self._chunk_dict[did] = shape
        else:
            self._chunk_dict[did] = chunk
        return self

    def add_full(self, vid: str, value, dtype):
        """
        Adds a constant array to the dataset.

        :param vid: The variable identifier.
        :param value: The value of the constant.
        :param dtype: The data type of the array.
        :return: The builder itself.
        """
        assert (
            vid not in self._array_dict
        ), f"Variable ID '{vid}' is already associated with an array"
        assert (
            vid in self._dimid_dict
        ), f"Variable ID '{vid}' is not associated with dimensions"
        did = self._dimid_dict[vid]
        array = da.full(
            self._shape(did),
            value,
            dtype=dtype,
            chunks=self._chunk_sizes(did),
        )
        self.add_array(vid, array)
        return self

    def build(self) -> Dataset:
        """
        Builds the dataset.

        :return: The dataset.
        """
        c_vars = {}
        d_vars = {}
        for vid, array in self._array_dict.items():
            array = DataArray(
                data=array, dims=self._dimid_dict[vid], name=vid
            )
            if vid in self._shape_dict:  # convention
                c_vars[vid] = array
            else:
                d_vars[vid] = array
        return Dataset(
            data_vars=d_vars, coords=c_vars, attrs=self._attrs_dict
        )

    def clear(self):
        """Releases all resources."""
        self._attrs_dict.clear()
        self._array_dict.clear()
        self._dimid_dict.clear()
        self._shape_dict.clear()
        self._chunk_dict.clear()

    def _chunk_size(self, did: str) -> int:
        """This method does not belong to public API."""
        if did in self._chunk_dict:
            chunk = self._chunk_dict[did]
        else:
            chunk = self._shape_dict[did]
        return chunk

    def _chunk_sizes(self, did: Any) -> tuple[int, ...]:
        """This method does not belong to public API."""
        return (
            tuple([self._chunk_size(d) for d in did])
            if isinstance(did, tuple)
            else (self._chunk_size(did),)
        )

    def _shape(self, did: Any) -> tuple[int, ...]:
        """This method does not belong to public API."""
        return (
            tuple([self._shape_dict[d] for d in did])
            if isinstance(did, tuple)
            else (self._shape_dict[did],)
        )

    def _assert_did_is_defined(self, did: str, var: str):
        """This method does not belong to public API."""
        assert (
            did in self._shape_dict
        ), f"Dimension ID '{did}' is not defined in {var}"

    def _assert_did_is_undefined(self, did: str):
        """This method does not belong to public API."""
        assert (
            did not in self._shape_dict
        ), f"Dimension ID '{did}' is already defined"

    def _assert_vid_is_undefined(self, vid: str):
        """This method does not belong to public API."""
        assert (
            vid not in self._dimid_dict
        ), f"Variable ID '{vid}' is already associated with dimensions"
        assert (
            vid not in self._array_dict
        ), f"Variable ID '{vid}' is already associated with an array"
