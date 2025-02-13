#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This code produces plots to create a movie.
"""

import warnings
from argparse import ArgumentDefaultsHelpFormatter
from argparse import ArgumentParser
from numbers import Number
from pathlib import Path

import numpy as np
from matplotlib import colors as plc
from xarray import DataArray

from wqf.algorithms.gaussian import Gaussian
from wqf.interface.constants import DID_TIM
from wqf.readerfactory import ReaderFactory
from wqf.val.benchmarks import BGC
from wqf.val.benchmarks import XGB
from wqf.val.period import Period
from wqf.val.plots import ScenePlot

warnings.filterwarnings(
    "ignore",
)


def plot_analysis(chl: DataArray, period: Period):
    chl = period.slice(chl)

    for t in range(chl.shape[0]):
        time = np.datetime_as_string(chl.coords[DID_TIM].values[t], unit="D")
        file = f"an___{time}"
        if Path(f"{file}.png").exists():
            continue
        ScenePlot().plot(
            chl[t],
            title=f"(Re)analysis {time}",
            fn=file,
            cbar_label=r"chlorophyll concentration (mg m$^{-3}$)",
            norm=plc.SymLogNorm(1.0, linscale=0.1, vmin=0.0, vmax=100.0),
            xlocs=(1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0),
            vmin=0.0,
            vmax=100.0,
        ).clear()


def plot_forecast(
    chl: DataArray, period: Period, fwhm: Number | None, h: int
):
    chl = period.slice(chl)

    if fwhm is not None:
        g = Gaussian(chl.dtype)
        chl = DataArray(
            data=g.apply_to(chl.data, fwhm=fwhm),
            coords=chl.coords,
            dims=chl.dims,
            attrs=chl.attrs,
        )

    for t in range(chl.shape[0]):
        time = np.datetime_as_string(chl.coords[DID_TIM].values[t], unit="D")
        file = f"fc_{h}_{time}"
        if Path(f"{file}.png").exists():
            continue
        ScenePlot().plot(
            chl[t],
            title=f"{h}-day forecast {time}",
            fn=file,
            cbar_label=r"chlorophyll concentration (mg m$^{-3}$)",
            norm=plc.SymLogNorm(1.0, linscale=0.1, vmin=0.0, vmax=100.0),
            xlocs=(1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0),
            vmin=0.0,
            vmax=100.0,
        ).clear()


def plot_observed(chl: DataArray, period: Period, fwhm: Number | None):
    chl = period.slice(chl)

    if fwhm is not None:
        g = Gaussian(chl.dtype)
        chl = DataArray(
            data=g.apply_to(chl.data, fwhm=fwhm),
            coords=chl.coords,
            dims=chl.dims,
            attrs=chl.attrs,
        )

    for t in range(chl.shape[0]):
        time = np.datetime_as_string(chl.coords[DID_TIM].values[t], unit="D")
        file = f"in___{time}"
        if Path(f"{file}.png").exists():
            continue
        ScenePlot().plot(
            chl[t],
            title=f"Observed {time}",
            fn=file,
            cbar_label=r"chlorophyll concentration (mg m$^{-3}$)",
            norm=plc.SymLogNorm(1.0, linscale=0.1, vmin=0.0, vmax=100.0),
            xlocs=(1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0),
            vmin=0.0,
            vmax=100.0,
        ).clear()


def generate_figures(args):
    reader = ReaderFactory.create_reader(args.aws)
    period = Period(args.period_start, args.period_end)

    obs = reader.read(args.cube_id, depth_level=3.0)
    ref, pre = XGB(args).predict(obs)
    plot_observed(ref, period=period, fwhm=args.fwhm)
    pre.close()
    ref.close()
    obs.close()

    if args.analysis:
        obs = reader.read(args.cube_id, depth_level=3.0)
        ref, pre = BGC(args).predict(obs)
        plot_analysis(pre, period=period)
        pre.close()
        ref.close()
        obs.close()

    for h in [1, 2, 3, 4, 5, 6, 7]:
        obs = reader.read(args.cube_id, depth_level=3.0)
        ref, pre = XGB(args).predict(obs, h=h)
        plot_forecast(pre, period=period, fwhm=args.fwhm, h=h)
        pre.close()
        ref.close()
        obs.close()


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="wqf-mov",
        description="This command produces figures to create a movie.",
        epilog="Copyright (c) Brockmann Consult GmbH, 2024. License: MIT",
        exit_on_error=True,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("cube_id", help="the data cube identifier")
    parser.add_argument("bgcm_id", help="the BGCM data identifier")
    parser.add_argument("xgbm_id", help="the XGBM data identifier")
    parser.add_argument(
        "--aws",
        help="read from AWS team store",
        action="store_true",
        default=False,
        required=False,
        dest="aws",
    )
    parser.add_argument(
        "--analysis",
        help="plot BGCM (re)analysis",
        action="store_true",
        default=True,
        required=False,
        dest="analysis",
    )
    parser.add_argument(
        "--no-analysis",
        help="do not plot BGCM (re)analysis",
        action="store_false",
        default=False,
        required=False,
        dest="analysis",
    )
    parser.add_argument(
        "--gaussian-filter",
        help="specify the full width at half maximum (pixels) of "
        "a lateral Gaussian filter applied to the forecast. "
        "If not specified no filter is applied.",
        type=float,
        required=False,
        dest="fwhm",
    )
    parser.add_argument(
        "--period-start",
        help="specify the first year of the considered period.",
        type=int,
        default=2020,
        required=False,
        dest="period_start",
    )
    parser.add_argument(
        "--period-end",
        help="specify the last year of the considered period.",
        type=int,
        default=2020,
        required=False,
        dest="period_end",
    )
    generate_figures(parser.parse_args())
