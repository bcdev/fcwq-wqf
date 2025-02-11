#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This module defines several functions for plotting data.
"""
from typing import Any

import dask.array as da
import numpy as np
from cartopy.crs import LambertConformal
from cartopy.crs import PlateCarree
from cartopy.crs import Projection
from cartopy.feature import NaturalEarthFeature
from matplotlib import colors as plc
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from xarray import DataArray
from xarray import Dataset

from ..interface.plot import Plot

_LAND = NaturalEarthFeature(
    "physical", "land", "10m", edgecolor="face", facecolor="#e8e5db"
)


class ScenePlot(Plot):
    """
    A scene plot, usually for plotting the values for a single time step
    of a variable in a data cube or of a quantity derived thereof.
    """

    def plot(
        self,
        data: DataArray,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xlim: tuple[Any, Any] | None = None,
        ylim: tuple[Any, Any] | None = None,
        title: str | None = None,
        fn: str | None = None,
        show: bool = False,
        *,
        cbar_label: str | None = None,
        cmap: str = "viridis",
        norm: plc.Normalize | None = None,
        projection: Projection | None = None,
        xlocs: tuple[Any, ...] | None = None,
        ylocs: tuple[Any, ...] | None = None,
        vmin: Any | None = None,
        vmax: Any | None = None,
    ) -> Figure:
        if projection is None:
            mid_lat = round(data.lat.mean().item(), ndigits=1)
            mid_lon = round(data.lon.mean().item(), ndigits=1)
            max_lat = np.ceil(data.lat.max()).item()
            min_lat = np.floor(data.lat.min()).item()
            projection = LambertConformal(
                central_latitude=mid_lat,
                central_longitude=mid_lon,
                standard_parallels=(min_lat, max_lat),
            )
        fig, ax = plt.subplots(
            subplot_kw={"projection": projection},
        )
        cbar_kwargs = {}
        if cbar_label is not None:
            cbar_kwargs["label"] = cbar_label
        data.plot(
            ax=ax,
            x="lon",
            y="lat",
            robust=True,
            transform=PlateCarree(),
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            cmap=cmap,
            cbar_kwargs=cbar_kwargs,
        )
        ax.add_feature(_LAND)
        ax.autoscale_view()
        ax.gridlines(
            alpha=0.1,
            draw_labels={"bottom": "x", "left": "y"},
            x_inline=False,
            y_inline=False,
            xlocs=xlocs,
            ylocs=ylocs,
        )
        if title is not None:
            ax.set_title(title)
        if fn is not None:
            fig.savefig(f"{fn}.png", bbox_inches="tight", dpi=300)
        if show:
            fig.show()
        plt.close()
        return fig


class HistogramPlot(Plot):
    """
    A one-dimensional density plot, usually for plotting the distribution
    of the values of a variable in a data cube or a slice thereof or of a
    variable derived thereof.
    """

    def plot(
        self,
        data: DataArray,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xlim: tuple[Any, Any] | None = None,
        ylim: tuple[Any, Any] | None = None,
        title: str | None = None,
        fn: str | None = None,
        show: bool = False,
        *,
        bins: int | None = None,
        density: bool = False,
        log: bool = False,
        hist_range: tuple[Any, Any] | None = None,
    ) -> Figure:
        fig, ax = plt.subplots()
        data.plot.hist(
            ax=ax, bins=bins, range=hist_range, density=density, log=log
        )
        decorate(ax, xlabel, ylabel, xlim, ylim, title)
        if fn is not None:
            fig.savefig(f"{fn}.pdf")
        if show:
            fig.show()
        plt.close()
        return fig


class DensityPlot(Plot):
    """
    A two-dimensional density plot, usually for plotting the distribution
    of the values of two variables in a data cube or a slice thereof or of
    values derived thereof.
    """

    def plot(
        self,
        data: tuple[DataArray, DataArray],
        xlabel: str | None = None,
        ylabel: str | None = None,
        xlim: tuple[Any, Any] | None = None,
        ylim: tuple[Any, Any] | None = None,
        title: str | None = None,
        fn: str | None = None,
        show: bool = False,
        *,
        bins: tuple[int, int] | None = None,
        cbar_label: str | None = None,
        cmap: str | None = None,
        density: bool = True,
        hist_range: tuple[tuple[Any, Any], tuple[Any, Any]] | None = None,
        norm: plc.Normalize | None = None,
        vmin: Any | None = None,
        vmax: Any | None = None,
    ) -> Figure:
        fig, ax = plt.subplots()
        cbar_kwargs = {}
        if cbar_label is not None:
            cbar_kwargs["label"] = cbar_label
        hist(data, bins, hist_range, density).plot(
            ax=ax,
            x="x",
            y="y",
            robust=True,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            cmap=cmap,
            cbar_kwargs=cbar_kwargs,
        )
        decorate(ax, xlabel, ylabel, xlim, ylim, title)
        if fn is not None:
            fig.savefig(f"{fn}.png", dpi=300)
        if show:
            fig.show()
        plt.close()
        return fig


class ScatterPlot(Plot):
    """
    A scatter plot, usually for plotting random samples of the values of
    two variables in a data cube or a slice thereof or of values derived
    thereof.
    """

    def plot(
        self,
        data: tuple[DataArray, DataArray],
        xlabel: str | None = None,
        ylabel: str | None = None,
        xlim: tuple[Any, Any] | None = None,
        ylim: tuple[Any, Any] | None = None,
        title: str | None = None,
        fn: str | None = None,
        show: bool = False,
        *,
        point_alpha: Any | None = None,
        point_color: str | None = None,
        point_marker: str | None = ".",
        sample_count: int = 4000,
    ) -> Figure:
        fig, ax = plt.subplots()
        rand(data, sample_count).plot.scatter(
            ax=ax,
            x="x",
            y="y",
            alpha=point_alpha,
            c=point_color,
            marker=point_marker,
        )
        decorate(ax, xlabel, ylabel, xlim, ylim, title)
        if fn is not None:
            fig.savefig(f"{fn}.png", dpi=300)
        if show:
            fig.show()
        plt.close()
        return fig


class TimeSeriesPlot(Plot):
    """
    A time series plot.

    Time series data may be grouped by year or month and values are indicated
    by markers, which are not connected.
    """

    def plot(
        self,
        data: DataArray,
        xlabel: str | None = "time",
        ylabel: str | None = None,
        xlim: tuple[np.datetime64, np.datetime64] | None = (
            np.datetime64("2016-01-01"),
            np.datetime64("2021-01-01"),
        ),
        ylim: tuple | None = None,
        title: str | None = None,
        fn: str | None = None,
        show: bool = False,
        *,
        point_marker: str = ".",
        group_by: str | None = "time.month",
    ) -> Figure:
        fig, ax = plt.subplots()

        if group_by is not None:
            for _, period in time_series(data).groupby(group_by):
                period.plot.scatter(ax=ax, marker=point_marker)
        else:
            time_series(data).scatter(ax=ax, marker=point_marker)
        decorate(ax, xlabel, ylabel, xlim, ylim, title)
        if fn is not None:
            fig.savefig(f"{fn}.png", dpi=300)
        if show:
            fig.show()
        plt.close()
        return fig


def time_series(ts: DataArray):
    """
    Returns a properly labelled time series.

    When raw time data are decoded using CF time, the result is an
    array of CF time objects.

    This method shall be used if the time series date are labelled
    with CF time objects, which are not understood by the plotting
    method.
    """
    return DataArray(
        data=ts.data,
        coords={  # avoid referring to 'time' by name
            dim: np.array([np.datetime64(t) for t in ts.coords[dim].values])
            for dim in ts.dims[:1]  # 'time' is the first dimension
        },
        dims=ts.dims,
        attrs=ts.attrs,
    )


def coords(edges: da.Array) -> da.Array:
    """Returns the coordinate values for given bin edges."""
    return (edges[:-1] + edges[1:]) / 2.0


def decorate(
    ax: Axes, xlabel=None, ylabel=None, xlim=None, ylim=None, title=None
):
    """
    Decorates a plot.

    Adds axis labels and a plot title, and sets axis limits. If
    a plot is a cartographic map, land features and geographic
    coordinate labels and are added, too.
    """

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title is not None:
        ax.set_title(title)


def hist(
    data: tuple[DataArray, DataArray],
    bins: tuple[int, int] | None,
    hist_range: tuple[tuple[Any, Any], tuple[Any, Any]] | None,
    density: bool = False,
) -> DataArray:
    """
    Returns a two-dimensional histogram of the given data.

    :param data: The pair of x and y data.
    :param bins: The histogram bins.
    :param hist_range: The range of the histogram.
    :param density: If `False`, the histogram gives the number of samples for
    each bin. If `True`, the histogram shows the probability density function
    for each bin.
    """
    assert (
        data[0].chunks == data[1].chunks
    ), f"chunks do not match: {data[0].chunks} != {data[1].chunks}"

    x: da.Array = _x(data)
    y: da.Array = _y(data)

    assert (
        x.chunks == y.chunks
    ), f"chunks do not match: {x.chunks} != {y.chunks}"

    h, x, y = da.histogram2d(
        x, y, bins=bins, range=hist_range, density=density
    )
    return DataArray(
        data=h,
        coords=[
            (DataArray(data=coords(x), dims="x")),
            (DataArray(data=coords(y), dims="y")),
        ],
        dims=["x", "y"],
    )


def rand(data: [DataArray, DataArray], sample_count: int) -> Dataset:
    """
    Returns a dataset of (x, y) samples randomly drawn from the data
    supplied as argument.

    :param data: The x and y data.
    :param sample_count: The number of random samples to draw.
    :return: The dataset of random samples.
    """
    x: da.Array = _x(data)
    y: da.Array = _y(data)
    i: da.Array = da.random.randint(low=0, high=x.size, size=sample_count)
    return Dataset(
        data_vars={
            "x": DataArray(data=x[i].compute(), dims="sample_count"),
            "y": DataArray(data=y[i].compute(), dims="sample_count"),
        }
    )


def _x(data: tuple[DataArray, DataArray]) -> da.Array:
    """Returns the x data."""
    return data[0].data.ravel()


def _y(data: tuple[DataArray, DataArray]) -> da.Array:
    """Returns the y data."""
    return data[1].data.ravel()
