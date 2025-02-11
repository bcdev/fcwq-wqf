#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This module provides unit-level tests for the WQF source product reader.
"""

import json
import unittest
import warnings
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from dask.array import Array
from xarray import DataArray
from xarray import Dataset

from wqf.interface.constants import VID_CHL
from wqf.interface.constants import VID_LAT
from wqf.interface.constants import VID_LON
from wqf.interface.constants import VID_TIM
from wqf.reader import Reader
from wqf.util.ncbin import ncgen

warnings.filterwarnings("ignore")


class ReaderTest(unittest.TestCase):
    """
    Tests the source product reader.

    Generates product files from CDL templates and reads them.
    """

    config: dict[str:Any]
    """
    The reader configuration.
    """
    files: list[Path]
    """
    The list of source product datasets generated from CDL
    template files.
    """

    def setUp(self):
        """
        Initializes the test.

        Generates netCDF datasets from source product template CDL files.
        """
        package = "wqf.config"
        name = "wqf.config.reader.json"
        with resources.path(package, name) as resource:
            with open(resource) as r:
                self.config = json.load(r)
                self.reconfigure()
        self.generate_source_files()

    def generate_source_files(self):
        """Generate dummy source datasets from template CDL files."""
        self.files = []
        with resources.path("wqf.templates", "in.cdl") as resource:
            self.files.append(ncgen(resource))

    def reconfigure(self):
        """Reconfigures chunk sizes and reader engine for this test."""
        self.config["config.wqf.reader.chunks"]["time"] = -1
        self.config["config.wqf.reader.chunks"]["lat"] = 67
        self.config["config.wqf.reader.chunks"]["lon"] = 31
        self.config["config.wqf.reader.engine"] = "h5netcdf"

    def tearDown(self):
        """Cleans up the test."""
        for f in self.files:
            f.unlink()
        self.files.clear()

    def test_read(self):
        """Tests reading a generated source dataset."""
        reader = Reader(self.config)
        for f in self.files:
            ds = reader.read(f)
            self.assertIsInstance(ds, Dataset)

            global_attrs = ds.attrs
            self.assertIsInstance(global_attrs, dict)

            time = ds[VID_TIM]
            self.assertIsInstance(time, DataArray)

            lat = ds[VID_LAT]
            self.assertIsInstance(lat, DataArray)

            lon = ds[VID_LON]
            self.assertIsInstance(lon, DataArray)

            chl = ds[VID_CHL]
            self.assertIsInstance(chl, DataArray)
            self.assertEqual("mg m-3", chl.attrs["units"])
            self.assertIsInstance(chl.data, Array)
            self.assertEqual(8, chl.data.chunksize[0])
            self.assertEqual(67, chl.data.chunksize[1])
            self.assertEqual(31, chl.data.chunksize[2])
            self.assertTrue(np.isnan(chl[0, 0, 0]))

            ds.close()


if __name__ == "__main__":
    unittest.main()
