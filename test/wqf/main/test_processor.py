#  Copyright (c) European Union, 2024
#  License: Proprietary

"""This module provides integration test cases."""

import unittest
from abc import ABC
from abc import abstractmethod
from importlib import resources
from pathlib import Path

import dask.array as da
import numpy as np
from xarray import Dataset

from wqf.interface.constants import DID_LAT
from wqf.interface.constants import DID_LON
from wqf.interface.constants import VID_CHL
from wqf.interface.constants import VID_TIM
from wqf.interface.exitcodes import ExitCodes
from wqf.main import main
from wqf.reader import Reader
from wqf.val.plots import ScenePlot
from wqf.val.plots import TimeSeriesPlot


class NoArgumentsTest(unittest.TestCase):
    """Tests a call without command line arguments."""

    def test_run_without_arguments(self):
        """
        Tests missing command line arguments. Termination with
        nonzero exit code is expected.
        """
        exit_code = main.run()
        self.assertEqual(ExitCodes.FAILURE_ARGUMENT_ERROR, exit_code)


class HelpTest(unittest.TestCase):
    """Tests the `-h` and `--help` command line options."""

    def test_option_h(self):
        """Tests the `-h` command line option."""

        exit_code = main.run(args=["-h"])
        self.assertEqual(ExitCodes.SUCCESS, exit_code)

    def test_option_help(self):
        """Tests the `--help` command line option."""
        exit_code = main.run(args=["--help"])
        self.assertEqual(ExitCodes.SUCCESS, exit_code)


class VersionTest(unittest.TestCase):
    """Tests the `-v` and `--version` command line options."""

    def test_option_v(self):
        """Tests the `-v` command line option."""
        exit_code = main.run(args=["-v"])
        self.assertEqual(ExitCodes.SUCCESS, exit_code)

    def test_option_version(self):
        """Tests the `--version` command line option."""
        exit_code = main.run(args=["--version"])
        self.assertEqual(ExitCodes.SUCCESS, exit_code)


class AbstractProcessingTest(ABC, unittest.TestCase):
    """
    Tests the processing of a test dataset.

    Target datasets are stored in netCDF for convenience.
    """

    @property
    @abstractmethod
    def args(self) -> list[str]:
        """The command line arguments."""

    @property
    @abstractmethod
    def model(self) -> str:
        """The model name."""

    @property
    @abstractmethod
    def plot(self) -> bool:
        """To generate plots."""

    def tearDown(self):
        """Cleans up the test."""
        self.target_file.unlink(missing_ok=True)

    @property
    def source_file(self) -> Path:
        """Returns the path to the source file."""
        return Path(self.args[-2])

    @property
    def target_file(self) -> Path:
        """Returns the path to the target file."""
        return Path(self.args[-1])

    def test_processing(self):
        """
        Processes the source dataset and compares the target dataset
        to the expected result.
        """
        self.assert_target_file_does_not_exist()

        exit_code = main.run(args=self.args)
        self.assertEqual(0, exit_code)

        self.assert_target_file_exists()
        source: Dataset = open_dataset(self.source_file)
        target: Dataset = open_dataset(self.target_file)
        expect: Dataset = open_dataset(resource(f"{self.model}.nc"))
        try:
            self.assert_shapes(expect, target)
            self.assert_almost_no_difference(expect, target)

            time_stamps = target[VID_TIM].values
            # noinspection PyTypeChecker
            self.assertEqual(
                np.datetime64("2020-07-06"),
                np.datetime64(time_stamps[0]),
            )
            # noinspection PyTypeChecker
            self.assertEqual(
                np.datetime64("2020-07-07"),
                np.datetime64(time_stamps[1]),
            )
            # noinspection PyTypeChecker
            self.assertEqual(
                np.datetime64("2020-07-08"),
                np.datetime64(time_stamps[2]),
            )
            if self.plot:
                for h in range(3, 0, -1):
                    plot_ds(source, h, f"Observed (day {h})", f"test_in_{h}")
                    plot_ds(
                        target, h, f"Forecast (day {h})", f"{self.model}_{h}"
                    )
                plot_ts(source, "Observed", "test_in")
                plot_ts(target, "Forecast", f"{self.model}")
        finally:
            target.close()
            expect.close()
            source.close()

    def assert_almost_no_difference(
        self, expect: Dataset, actual: Dataset, delta=0.0, q=1.0
    ):
        """This method does not belong to public API."""
        for name, _ in expect.data_vars.items():
            a: np.ndarray = actual[name].values
            b: np.ndarray = expect[name].values
            d = np.nanquantile(np.abs(a - b), q)
            self.assertAlmostEqual(0.0, d, delta=delta, msg=f"{name}: {d}")

    def assert_shapes(self, expect: Dataset, actual: Dataset):
        """This method does not belong to public API."""
        for name, _ in expect.variables.items():
            a: da.Array = actual[name].data
            b: da.Array = expect[name].data
            self.assertEqual(b.shape, a.shape, msg=f"{name}: {b}, {a}")

    def assert_target_file_exists(self):
        """This method does not belong to public API."""
        self.assertTrue(self.target_file.exists())

    def assert_target_file_does_not_exist(self):
        """This method does not belong to public API."""
        self.assertFalse(self.target_file.exists())


class CentralProcessingTest(AbstractProcessingTest):
    """Tests the central North Sea model."""

    @property
    def args(self) -> list[str]:
        source_file = resource("test_in.nc")
        target_file = f"{self.model}.nc"
        return [
            "--horizon",
            "3",
            "--log-level",
            "warning",
            "--model",
            f"ns-{self.model}",
            f"{source_file}",
            f"{target_file}",
        ]

    @property
    def model(self) -> str:
        return "central"

    @property
    def plot(self) -> bool:
        return False


class CoastalProcessingTest(AbstractProcessingTest):
    """Tests the coastal North Sea model."""

    @property
    def args(self) -> list[str]:
        source_file = resource("test_in.nc")
        target_file = f"{self.model}.nc"
        return [
            "--horizon",
            "3",
            "--log-level",
            "warning",
            "--model",
            f"ns-{self.model}",
            f"{source_file}",
            f"{target_file}",
        ]

    @property
    def model(self) -> str:
        return "coastal"

    @property
    def plot(self) -> bool:
        return False


class NaturalProcessingTest(AbstractProcessingTest):
    """Tests the natural North Sea model."""

    @property
    def args(self) -> list[str]:
        source_file = resource("test_in.nc")
        target_file = f"{self.model}.nc"
        return [
            "--horizon",
            "3",
            "--log-level",
            "warning",
            "--model",
            f"ns-{self.model}",
            f"{source_file}",
            f"{target_file}",
        ]

    @property
    def model(self) -> str:
        return "natural"

    @property
    def plot(self) -> bool:
        return False


def plot_ts(ts: Dataset, title: str, filename: str):
    """Generates a time series plot."""
    TimeSeriesPlot().plot(
        ts[VID_CHL].mean([DID_LAT, DID_LON]),
        xlim=(np.datetime64("2020-07-01"), np.datetime64("2020-07-10")),
        ylim=(0.0, 3.0),
        xlabel="day",
        ylabel=r"chlorophyll concentration (mg m$^{-3}$)",
        title=title,
        fn=filename,
    )


def plot_ds(ds: Dataset, h, title: str, filename: str):
    """Generates a scene plot."""
    import matplotlib.colors as plc

    ScenePlot().plot(
        ds[VID_CHL][h - 4],
        title=title,
        fn=filename,
        cbar_label=r"chlorophyll concentration (mg m$^{-3}$)",
        norm=plc.LogNorm(),
        vmin=1.0,
        vmax=100.0,
        xlocs=(1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0),
    )


def open_dataset(path: Path) -> Dataset:
    """This method does not belong to public API."""
    return Reader().read(path)


def resource(name: str) -> Path:
    """This method does not belong to public API."""
    with resources.path("test.wqf.resources", name) as _:
        return _


if __name__ == "__main__":
    unittest.main()
