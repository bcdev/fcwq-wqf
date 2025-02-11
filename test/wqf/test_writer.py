#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This module provides unit-level tests for the WQF target product writer.
"""

import json
import unittest
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

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


class WriterTest(unittest.TestCase):
    """Tests the WQF target product writer."""

    config: dict[str:Any]
    """The writer configuration."""

    def setUp(self):
        """Initializes the test."""
        package = "wqf.config"
        name = "wqf.config.writer.json"
        with resources.path(package, name) as resource:
            with open(resource) as r:
                self.config = json.load(r)

    def test_write(self):
        """Tests writing a generated WQF target dataset."""
        file = Path("fc.nc")
        file.unlink(missing_ok=True)

        writer = Writer(self.config, engine="h5netcdf")
        writer.write(self.dataset_builder.build(), file)
        self.assertTrue(file.is_file())

        dump = ncdump(file)
        self.assertTrue(dump.is_file())
        dump.unlink()
        file.unlink()

    @property
    def dataset_builder(self):
        shape: tuple[int, int, int] = 3, 469, 527
        chunk: tuple[int, int, int] = -1, 67, 31
        builder = DatasetBuilder()
        builder.add_dim(DID_TIM, shape[0], chunk[0])
        builder.add_dim(DID_LAT, shape[1], chunk[1])
        builder.add_dim(DID_LON, shape[2], chunk[2])
        builder.add_var(VID_TIM, DID_TIM)
        builder.add_var(VID_LAT, DID_LAT)
        builder.add_var(VID_LON, DID_LON)
        builder.add_var(VID_CHL, (DID_TIM, DID_LAT, DID_LON))
        builder.add_full(VID_TIM, 0.0, np.double)
        builder.add_full(VID_LAT, 0.0, np.double)
        builder.add_full(VID_LON, 0.0, np.double)
        builder.add_full(VID_CHL, 0.0, np.single)
        return builder


if __name__ == "__main__":
    unittest.main()
