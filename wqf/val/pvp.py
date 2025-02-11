#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This module defines plots corresponding to the Product Validation Plan (PVP).
"""
import warnings
from argparse import ArgumentDefaultsHelpFormatter
from argparse import ArgumentParser
from typing import Any

import dask.array as da
import numpy as np
from matplotlib import colors as plc
from matplotlib.figure import Figure
from xarray import DataArray

from wqf.interface.constants import VID_CHL
from wqf.readerfactory import ReaderFactory
from wqf.val.benchmarks import Naive
from wqf.val.metrics import Bias
from wqf.val.metrics import Count
from wqf.val.metrics import MAD
from wqf.val.metrics import MAPD
from wqf.val.metrics import R2
from wqf.val.metrics import RMSE
from wqf.val.metrics import WRMSSE
from wqf.val.plots import DensityPlot
from wqf.val.plots import HistogramPlot
from wqf.val.plots import ScatterPlot
from wqf.val.plots import ScenePlot
from wqf.val.plots import TimeSeriesPlot

warnings.filterwarnings(
    "ignore",
)


def plot_bias_scene(
    data: DataArray, title: str | None = None, fn: str | None = None
) -> Figure:
    return ScenePlot().plot(
        data,
        title=title,
        fn=fn,
        cbar_label=r"bias (mg m$^{-3}$)",
        cmap="cividis",
        vmin=-1.5,
        vmax=1.5,
        xlocs=(1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0),
    )


def plot_bias_time_series(
    data: DataArray,
    title: str | None = None,
    fn: str | None = None,
    xlim: tuple[int, int] | None = None,
    ylim: tuple[Any, Any] = (-1.5, 1.5),
) -> Figure:
    return TimeSeriesPlot().plot(
        data,
        ylabel=r"bias (mg m$^{-3}$)",
        xlim=(
            (
                np.datetime64(f"{xlim[0]}-01-01"),
                np.datetime64(f"{xlim[1]}-01-01"),
            )
            if xlim is not None
            else None
        ),
        ylim=ylim,
        title=title,
        fn=fn,
    )


def plot_count_scene(
    data: DataArray, title: str | None = None, fn: str | None = None
) -> Figure:
    return ScenePlot().plot(
        data,
        title=title,
        fn=fn,
        cbar_label="number of forecasts",
        vmin=0.0,
        vmax=250.0,
        xlocs=(1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0),
    )


def plot_determination_coefficient_scene(
    data: DataArray, title: str | None = None, fn: str | None = None
) -> Figure:
    return ScenePlot().plot(
        data,
        title=title,
        fn=fn,
        cbar_label="coefficient of determination",
        vmin=0.0,
        vmax=1.0,
        xlocs=(1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0),
    )


def plot_determination_coefficient_time_series(
    data: DataArray,
    title: str | None = None,
    fn: str | None = None,
    xlim: tuple[int, int] | None = None,
    ylim: tuple[Any, Any] = (-1.0, 1.0),
) -> Figure:
    return TimeSeriesPlot().plot(
        data,
        ylabel="coefficient of determination",
        xlim=(
            (
                np.datetime64(f"{xlim[0]}-01-01"),
                np.datetime64(f"{xlim[1]}-01-01"),
            )
            if xlim is not None
            else None
        ),
        ylim=ylim,
        title=title,
        fn=fn,
    )


def plot_mad_scene(
    data: DataArray, title: str | None = None, fn: str | None = None
) -> Figure:
    return ScenePlot().plot(
        data,
        title=title,
        fn=fn,
        cbar_label=r"MAD (mg m$^{-3}$)",
        vmin=1.0,
        vmax=7.0,
        xlocs=(1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0),
    )


def plot_mad_time_series(
    data: DataArray,
    title: str | None = None,
    fn: str | None = None,
    xlim: tuple[int, int] | None = None,
    ylim: tuple[Any, Any] = (0.0, 6.0),
) -> Figure:
    return TimeSeriesPlot().plot(
        data,
        ylabel=r"MAD (mg m$^{-3}$)",
        xlim=(
            (
                np.datetime64(f"{xlim[0]}-01-01"),
                np.datetime64(f"{xlim[1]}-01-01"),
            )
            if xlim is not None
            else None
        ),
        ylim=ylim,
        title=title,
        fn=fn,
    )


def plot_mapd_scene(
    data: DataArray, title: str | None = None, fn: str | None = None
) -> Figure:
    return ScenePlot().plot(
        data,
        title=title,
        fn=fn,
        cbar_label=r"MAPD ($10^2$)",
        vmin=0.0,
        vmax=1.0,
        xlocs=(1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0),
    )


def plot_mapd_time_series(
    data: DataArray,
    title: str | None = None,
    fn: str | None = None,
    xlim: tuple[int, int] | None = None,
    ylim: tuple[Any, Any] = (0.0, 1.0),
) -> Figure:
    return TimeSeriesPlot().plot(
        data,
        ylabel=r"MAPD ($10^2$)",
        xlim=(
            (
                np.datetime64(f"{xlim[0]}-01-01"),
                np.datetime64(f"{xlim[1]}-01-01"),
            )
            if xlim is not None
            else None
        ),
        ylim=ylim,
        title=title,
        fn=fn,
    )


def plot_rmse_scene(
    data: DataArray, title: str | None = None, fn: str | None = None
) -> Figure:
    return ScenePlot().plot(
        data,
        title=title,
        fn=fn,
        cbar_label=r"RMSE (mg m$^{-3}$)",
        vmin=1.0,
        vmax=7.0,
        xlocs=(1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0),
    )


def plot_rmse_time_series(
    data: DataArray,
    title: str | None = None,
    fn: str | None = None,
    xlim: tuple[int, int] | None = None,
    ylim: tuple[Any, Any] = (0.0, 7.0),
) -> Figure:
    return TimeSeriesPlot().plot(
        data,
        ylabel=r"RMSE (mg m$^{-3}$)",
        xlim=(
            (
                np.datetime64(f"{xlim[0]}-01-01"),
                np.datetime64(f"{xlim[1]}-01-01"),
            )
            if xlim is not None
            else None
        ),
        ylim=ylim,
        title=title,
        fn=fn,
    )


def plot_wrmsse_scene(
    data: DataArray, title: str | None = None, fn: str | None = None
) -> Figure:
    return ScenePlot().plot(
        data,
        title=title,
        fn=fn,
        cbar_label="WRMSSE",
        vmin=1.0,
        vmax=7.0,
        xlocs=(1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0),
    )


def plot_wrmsse_time_series(
    data: DataArray,
    title: str | None = None,
    fn: str | None = None,
    xlim: tuple[int, int] | None = None,
    ylim: tuple[Any, Any] = (0.0, 7.0),
) -> Figure:
    return TimeSeriesPlot().plot(
        data,
        ylabel="WRMSSE",
        xlim=(
            (
                np.datetime64(f"{xlim[0]}-01-01"),
                np.datetime64(f"{xlim[1]}-01-01"),
            )
            if xlim is not None
            else None
        ),
        ylim=ylim,
        title=title,
        fn=fn,
    )


def plot_value_density(
    data: tuple[DataArray, DataArray],
    title: str | None = None,
    fn: str | None = None,
) -> Figure:
    return DensityPlot().plot(
        data,
        xlabel=r"reference chlorophyll concentration (mg m$^{-3}$)",
        ylabel=r"forecast (mg m$^{-3}$)",
        title=title,
        fn=fn,
        bins=(30, 30),
        cbar_label=r"probability density (m$^{6}$ mg$^{-2}$)",
        hist_range=((0.0, 30.0), (0.0, 30.0)),
        norm=plc.SymLogNorm(0.0001, vmin=0.0, vmax=1.0),
    )


def plot_value_scatter(
    data: tuple[DataArray, DataArray],
    title: str | None = None,
    fn: str | None = None,
) -> Figure:
    return ScatterPlot().plot(
        data,
        xlabel=r"reference chlorophyll concentration (mg m$^{-3}$)",
        ylabel=r"forecast (mg m$^{-3}$)",
        xlim=(-0.50, 30.5),
        ylim=(-0.50, 30.5),
        title=title,
        fn=fn,
    )


def plot_error_density(
    data: tuple[DataArray, DataArray],
    title: str | None = None,
    fn: str | None = None,
) -> Figure:
    return DensityPlot().plot(
        data,
        xlabel=r"reference chlorophyll concentration (mg m$^{-3}$)",
        ylabel=r"forecast error (mg m$^{-3}$)",
        title=title,
        fn=fn,
        bins=(30, 31),
        cbar_label=r"probability density (m$^{6}$ mg$^{-2}$)",
        hist_range=((0.0, 30.0), (-15.5, 15.5)),
        norm=plc.SymLogNorm(0.0001, vmin=0.0, vmax=1.0),
    )


def plot_error_scatter(
    data: tuple[DataArray, DataArray],
    title: str | None = None,
    fn: str | None = None,
) -> Figure:
    return ScatterPlot().plot(
        data,
        xlabel=r"reference chlorophyll concentration (mg m$^{-3}$)",
        ylabel=r"forecast error (mg m$^{-3}$)",
        xlim=(-0.50, 30.5),
        ylim=(-15.5, 15.5),
        title=title,
        fn=fn,
    )


def plot_error_histogram(
    data: DataArray,
    title: str | None = None,
    fn: str | None = None,
) -> Figure:
    return HistogramPlot().plot(
        data,
        xlabel=r"forecast error (mg m$^{-3}$)",
        ylabel=r"probability density (m$^{3}$ mg$^{-1}$)",
        ylim=(0.0, 1.0),
        title=title,
        fn=fn,
        bins=31,
        density=True,
        hist_range=(-15.5, 15.5),
    )


def plot_relative_error_density(
    data: tuple[DataArray, DataArray],
    title: str | None = None,
    fn: str | None = None,
) -> Figure:
    return DensityPlot().plot(
        data,
        xlabel=r"reference chlorophyll concentration (mg m$^{-3}$)",
        ylabel="forecast relative error",
        title=title,
        fn=fn,
        bins=(30, 31),
        cbar_label=r"probability density (m$^{3}$ mg$^{-1}$)",
        hist_range=((0.0, 30.0), (-1.55, 1.55)),
        norm=plc.SymLogNorm(0.0001, vmin=0.0, vmax=1.0),
    )


def plot_relative_error_scatter(
    data: tuple[DataArray, DataArray],
    title: str | None = None,
    fn: str | None = None,
) -> Figure:
    return ScatterPlot().plot(
        data,
        xlabel="reference chlorophyll concentration (mg m$^{-3}$)",
        ylabel="forecast relative error",
        xlim=(-0.50, 30.5),
        ylim=(-1.55, 1.55),
        title=title,
        fn=fn,
    )


def plot_relative_error_histogram(
    data: DataArray,
    title: str | None = None,
    fn: str | None = None,
) -> Figure:
    return HistogramPlot().plot(
        data,
        xlabel="forecast relative error",
        ylabel="probability density",
        ylim=(0.0, 2.0),
        title=title,
        fn=fn,
        bins=31,
        density=True,
        hist_range=(-1.55, 1.55),
    )


def generate_figures(args, period: tuple[int, int]):
    da.random.seed(42)

    reader = ReaderFactory.create_reader(args.aws)
    cube = reader.read(args.cube_id, depth_level=3.0)
    ref, pre = Naive().predict(cube[VID_CHL])

    plot_bias_time_series(
        Bias().series(ref, pre),
        "Mockup chlorophyll forecast",
        "bias_mockup_series",
        xlim=period,
    )
    plot_bias_scene(
        Bias().image(ref, pre),
        "Mockup chlorophyll forecast",
        "bias_mockup_image",
    )
    plot_error_histogram(
        Bias.err(ref, pre),
        "Mockup chlorophyll forecast",
        "err_mockup_hist",
    )
    plot_relative_error_histogram(
        Bias.rer(ref, pre, condition=ref > 1.0),
        "Mockup chlorophyll forecast",
        "rer_mockup_hist",
    )

    plot_count_scene(
        Count().image(ref, pre),
        "Mockup chlorophyll forecast",
        "count_mockup_image",
    )

    plot_determination_coefficient_time_series(
        R2().series(ref, pre),
        "Mockup chlorophyll forecast",
        "det_mockup_series",
        xlim=period,
    )
    plot_determination_coefficient_scene(
        R2().image(ref, pre),
        "Mockup chlorophyll forecast",
        "det_mockup_image",
    )

    plot_mad_time_series(
        MAD().series(ref, pre),
        "Mockup chlorophyll forecast",
        "mad_mockup_series",
        xlim=period,
    )
    plot_mad_scene(
        MAD().image(ref, pre),
        "Mockup chlorophyll forecast",
        "mad_mockup_image",
    )

    plot_mapd_time_series(
        MAPD().series(ref, pre, condition=ref > 1.0),
        "Mockup chlorophyll forecast",
        "mapd_mockup_series",
        xlim=period,
    )
    plot_mapd_scene(
        MAPD().image(ref, pre, condition=ref > 1.0),
        "Mockup chlorophyll forecast",
        "mapd_mockup_image",
    )

    plot_rmse_time_series(
        RMSE().series(ref, pre),
        "Mockup chlorophyll forecast",
        "rmse_mockup_series",
        xlim=period,
    )
    plot_rmse_scene(
        RMSE().image(ref, pre),
        "Mockup chlorophyll forecast",
        "rmse_mockup_image",
    )

    plot_wrmsse_time_series(
        WRMSSE().series(ref, pre, condition=ref > 1.0),
        "Mockup chlorophyll forecast",
        "wrmsse_mockup_series",
        xlim=period,
    )
    plot_wrmsse_scene(
        WRMSSE().image(ref, pre, condition=ref > 1.0),
        "Mockup chlorophyll forecast",
        "wrmsse_mockup_image",
    )

    x = ref.loc[f"{period[0]}-01-01" :f"{period[1]}-01-01", :, :]
    y = pre.loc[f"{period[0]}-01-01" :f"{period[1]}-01-01", :, :]

    plot_value_density(
        (x, y),
        "Mockup chlorophyll forecast",
        "val_mockup_density",
    )
    plot_value_scatter(
        (x, y),
        "Mockup chlorophyll forecast",
        "val_mockup_scatter",
    )
    plot_error_density(
        (x, y - x),
        "Mockup chlorophyll forecast",
        "err_mockup_density",
    )
    plot_error_scatter(
        (x, y - x),
        "Mockup chlorophyll forecast",
        "err_mockup_scatter",
    )
    plot_relative_error_density(
        (x, (y - x) / x),
        "Mockup chlorophyll forecast",
        "rer_mockup_density",
    )
    plot_relative_error_scatter(
        (x, (y - x) / x),
        "Mockup chlorophyll forecast",
        "rer_mockup_scatter",
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="wqf-pvp",
        description="This command produces figures for the Product "
        "Validation Plan (PVP).",
        epilog="Copyright (c) Brockmann Consult GmbH, 2024. License: MIT",
        exit_on_error=True,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("cube_id", help="the data cube identifier")
    parser.add_argument(
        "--aws",
        help="reading from AWS team store",
        action="store_true",
        default=False,
        required=False,
        dest="aws",
    )
    generate_figures(parser.parse_args(), (2016, 2021))
