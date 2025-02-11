#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This module provides code to generate a template of the FC-WQ target
product.
"""

import json
from importlib import resources
from pathlib import Path
from typing import Any
from typing import Literal

import dask.array as da
import numpy as np

from wqf import __version__
from wqf.datasetbuilder import DatasetBuilder
from wqf.interface.constants import DID_LAT
from wqf.interface.constants import DID_LON
from wqf.interface.constants import DID_TIM
from wqf.interface.constants import VID_CHL
from wqf.interface.constants import VID_LAT
from wqf.interface.constants import VID_LON
from wqf.interface.constants import VID_TIM
from wqf.util.ncbin import ncdump
from wqf.writer import Writer


class Generator:
    """
    Utility class to generate a template of the FC-WQ target
    product.
    """

    _config: dict[str:Any]

    def __init__(self, config: dict[str:Any]):
        """This method does not belong to public API."""
        self._config = config

    @staticmethod
    def create():
        """Creates a new generator instance."""
        package = "wqf.config"
        name = "wqf.config.writer.json"
        with resources.path(package, name) as resource:
            with open(resource) as r:
                config = json.load(r)
        return Generator(config)

    def generate(
        self,
        path: str | Path,
        shape: tuple[int, int, int] = (3, 469, 527),
        chunks: tuple[int, int, int] = (-1, 67, 31),
        engine: Literal["h5netcdf", "netcdf4", "zarr"] | None = None,
    ):
        """
        Generates and writes a template dataset.

        :param path: The path to the dataset file.
        :param shape: The shape of the dataset.
        :param chunks: The chunk sizes for time, latitude, and longitude
        dimensions, respectively.
        :param engine: The engine used to write the dataset.
        """
        Writer(self._config, engine=engine).write(
            self._create_dataset(shape, chunks), path
        )

    @staticmethod
    def _create_dataset(shape, chunks):
        builder = DatasetBuilder()
        builder.add_attr("version", __version__)

        builder.add_dim(DID_TIM, shape[0], chunks[0])
        builder.add_dim(DID_LAT, shape[1], chunks[1])
        builder.add_dim(DID_LON, shape[2], chunks[2])

        builder.add_var(VID_TIM, DID_TIM)
        builder.add_var(VID_LAT, DID_LAT)
        builder.add_var(VID_LON, DID_LON)
        builder.add_var(VID_CHL, (DID_TIM, DID_LAT, DID_LON))

        builder.add_array(
            VID_TIM,
            da.from_array(
                [
                    np.datetime64("2020-07-06T00:00", "ns")
                    + np.timedelta64(i, "D")  # noqa: W503
                    for i in range(shape[0])
                ],
            ),
        )
        builder.add_full(VID_LAT, 0.0, np.double)
        builder.add_full(VID_LON, 0.0, np.double)
        builder.add_full(VID_CHL, 0.0, np.single)
        return builder.build()


def _generate_cdl():
    ds_path = Path("fc.nc")
    ncdump(ds_path)


def _generate_nc():
    generator = Generator.create()
    ds_path = Path("fc.nc")
    generator.generate(ds_path, engine="h5netcdf")


def _generate_zarr():
    generator = Generator.create()
    ds_path = Path("fc.zarr")
    generator.generate(ds_path, engine="zarr")


if __name__ == "__main__":
    _generate_nc()
    _generate_zarr()
    _generate_cdl()
