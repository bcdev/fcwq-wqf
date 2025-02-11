#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This module produces plots for the Product Validation Report (PVR).
"""
import warnings
from argparse import ArgumentDefaultsHelpFormatter
from argparse import ArgumentParser

import dask.array as da
from xarray import DataArray

from wqf.interface.constants import VID_CHL
from wqf.readerfactory import ReaderFactory
from wqf.val.benchmarks import BGC
from wqf.val.benchmarks import MA
from wqf.val.benchmarks import Naive
from wqf.val.benchmarks import SES
from wqf.val.benchmarks import SNaive
from wqf.val.benchmarks import XGB
from wqf.val.metrics import Bias
from wqf.val.metrics import Count
from wqf.val.metrics import MAD
from wqf.val.metrics import MAPD
from wqf.val.metrics import R2
from wqf.val.metrics import RMSE
from wqf.val.metrics import WRMSSE
from wqf.val.period import Period
from wqf.val.pvp import plot_bias_scene
from wqf.val.pvp import plot_bias_time_series
from wqf.val.pvp import plot_count_scene
from wqf.val.pvp import plot_determination_coefficient_scene
from wqf.val.pvp import plot_determination_coefficient_time_series
from wqf.val.pvp import plot_error_density
from wqf.val.pvp import plot_error_histogram
from wqf.val.pvp import plot_error_scatter
from wqf.val.pvp import plot_mad_scene
from wqf.val.pvp import plot_mad_time_series
from wqf.val.pvp import plot_mapd_scene
from wqf.val.pvp import plot_mapd_time_series
from wqf.val.pvp import plot_relative_error_density
from wqf.val.pvp import plot_relative_error_histogram
from wqf.val.pvp import plot_relative_error_scatter
from wqf.val.pvp import plot_rmse_scene
from wqf.val.pvp import plot_rmse_time_series
from wqf.val.pvp import plot_value_density
from wqf.val.pvp import plot_value_scatter
from wqf.val.pvp import plot_wrmsse_scene
from wqf.val.pvp import plot_wrmsse_time_series

warnings.filterwarnings(
    "ignore",
)

_MIN_PIXELS = 30000
"""
The validation does not consider time steps (i.e., days) with less than
this number of cloud free pixels.
"""


def plot_bias_diagrams(
    ref: DataArray, pre: DataArray, method: str, period: Period
):
    ref = period.slice(ref)
    pre = period.slice(pre)

    value = Bias().value(ref, pre)
    print(
        f"Bias .............. ({method}) {period}: {value:6.2f}",
        flush=True,
    )
    value = Count().value(ref, pre)
    print(f"Count ............. ({method}) {period}: {value}", flush=True)

    title = f"{method} forecast {period}"
    plot_bias_time_series(
        Bias().series(ref, pre),
        title,
        f"bias_{method}_series_{period.str('_')}",
        xlim=period.lim,
    )
    plot_bias_scene(
        Bias().image(ref, pre),
        title,
        f"bias_{method}_image_{period.str('_')}",
    )
    plot_error_histogram(
        Bias().err(ref, pre),
        title,
        f"err_{method}_hist_{period.str('_')}",
    )
    plot_relative_error_histogram(
        Bias().rer(ref, pre, condition=ref > 1.0),
        title,
        f"rer_{method}_hist_{period.str('_')}",
    )
    plot_count_scene(
        Count().image(ref, pre),
        title,
        f"count_{method}_image_{period.str('_')}",
    )


def plot_det_coefficient_diagrams(
    ref: DataArray, pre: DataArray, method: str, period: Period
):
    ref = period.slice(ref)
    pre = period.slice(pre)

    value = R2().value(ref, pre)
    print(
        f"Coefficient of det. ({method}) {period}: {value:6.2f}",
        flush=True,
    )

    title = f"{method} forecast {period}"
    plot_determination_coefficient_time_series(
        R2().series(ref, pre),
        title,
        f"det_{method}_series_{period.str('_')}",
        xlim=period.lim,
    )
    plot_determination_coefficient_scene(
        R2().image(ref, pre),
        title,
        f"det_{method}_image_{period.str('_')}",
    )


def plot_mad_diagrams(
    ref: DataArray, pre: DataArray, method: str, period: Period
):
    ref = period.slice(ref)
    pre = period.slice(pre)

    value = MAD().value(ref, pre)
    print(
        f"MAD ............... ({method}) {period}: {value:6.2f}",
        flush=True,
    )

    title = f"{method} forecast {period}"
    plot_mad_time_series(
        MAD().series(ref, pre),
        title,
        f"mad_{method}_series_{period.str('_')}",
        xlim=period.lim,
    )
    plot_mad_scene(
        MAD().image(ref, pre),
        title,
        f"mad_{method}_image_{period.str('_')}",
    )


def plot_mapd_diagrams(
    ref: DataArray, pre: DataArray, method: str, period: Period
):
    ref = period.slice(ref)
    pre = period.slice(pre)

    value = MAPD().value(ref, pre, condition=ref > 1.0)
    print(
        f"MAPD .............. ({method}) {period}: {value:6.2f}",
        flush=True,
    )

    title = f"{method} forecast {period}"
    plot_mapd_time_series(
        MAPD().series(ref, pre, condition=ref > 1.0),
        title,
        f"mapd_{method}_series_{period.str('_')}",
        xlim=period.lim,
    )
    plot_mapd_scene(
        MAPD().image(ref, pre, condition=ref > 1.0),
        title,
        f"mapd_{method}_image_{period.str('_')}",
    )


def plot_rmse_diagrams(
    ref: DataArray, pre: DataArray, method: str, period: Period
):
    ref = period.slice(ref)
    pre = period.slice(pre)

    value = RMSE().value(ref, pre)
    print(
        f"RMSE .............. ({method}) {period}: {value:6.2f}",
        flush=True,
    )

    title = f"{method} forecast {period}"
    plot_rmse_time_series(
        RMSE().series(ref, pre),
        title,
        f"rmse_{method}_series_{period.str('_')}",
        xlim=period.lim,
    )
    plot_rmse_scene(
        RMSE().image(ref, pre),
        title,
        f"rmse_{method}_image_{period.str('_')}",
    )


def plot_wrmsse_diagrams(
    ref: DataArray, pre: DataArray, method: str, period: Period
):
    ref = period.slice(ref)
    pre = period.slice(pre)

    value = WRMSSE().value(ref, pre, condition=ref > 1.0)
    print(
        f"WRMSSE ............ ({method}) {period}: {value:6.2f}",
        flush=True,
    )

    title = f"{method} forecast {period}"
    plot_wrmsse_time_series(
        WRMSSE().series(ref, pre, condition=ref > 1.0),
        title,
        f"wrmsse_{method}_series_{period.str('_')}",
        xlim=period.lim,
    )
    plot_wrmsse_scene(
        WRMSSE().image(ref, pre, condition=ref > 1.0),
        title,
        f"wrmsse_{method}_image_{period.str('_')}",
    )


def plot_density_diagrams(
    ref: DataArray, pre: DataArray, method: str, period: Period
):
    x = period.slice(ref)
    y = period.slice(pre)

    plot_value_density(
        (x, y),
        f"{method} forecast {period}",
        f"val_{method}_density_{period.str('_')}",
    )
    plot_error_density(
        (x, y - x),
        f"{method} forecast {period}",
        f"err_{method}_density_{period.str('_')}",
    )
    plot_relative_error_density(
        (x, (y - x) / x),
        f"{method} forecast {period}",
        f"rer_{method}_density_{period.str('_')}",
    )
    plot_value_scatter(
        (x, y),
        f"{method} forecast {period}",
        f"val_{method}_scatter_{period.str('_')}",
    )
    plot_error_scatter(
        (x, y - x),
        f"{method} forecast {period}",
        f"err_{method}_scatter_{period.str('_')}",
    )
    plot_relative_error_scatter(
        (x, (y - x) / x),
        f"{method} forecast {period}",
        f"rer_{method}_scatter_{period.str('_')}",
    )


def plot_diagnostic_diagrams(ref: DataArray, pre: DataArray, method: str):
    plot_bias_diagrams(ref, pre, method, Period(2016, 2019))
    plot_bias_diagrams(ref, pre, method, Period(2020))

    plot_det_coefficient_diagrams(ref, pre, method, Period(2016, 2019))
    plot_det_coefficient_diagrams(ref, pre, method, Period(2020))

    plot_mad_diagrams(ref, pre, method, Period(2016, 2019))
    plot_mad_diagrams(ref, pre, method, Period(2020))

    plot_mapd_diagrams(ref, pre, method, Period(2016, 2019))
    plot_mapd_diagrams(ref, pre, method, Period(2020))

    plot_rmse_diagrams(ref, pre, method, Period(2016, 2019))
    plot_rmse_diagrams(ref, pre, method, Period(2020))

    plot_wrmsse_diagrams(ref, pre, method, Period(2016, 2019))
    plot_wrmsse_diagrams(ref, pre, method, Period(2020))

    plot_density_diagrams(ref, pre, method, Period(2016, 2019))
    plot_density_diagrams(ref, pre, method, Period(2020))


def generate_figures(args):
    da.random.seed(42)

    reader = ReaderFactory.create_reader(args.aws)
    cube = reader.read(args.cube_id, depth_level=3.0)

    for h in [7, 6, 5, 4, 3, 2, 1]:
        ref, xgb = XGB(args).predict(cube, min_pixels=_MIN_PIXELS, h=h)
        plot_diagnostic_diagrams(ref, xgb, f"XGB{h}")

    ref, pre = BGC(args).predict(cube, min_pixels=_MIN_PIXELS)
    plot_diagnostic_diagrams(ref, pre, "BGC")
    plot_diagnostic_diagrams(ref, xgb.where(pre.notnull()), "XGB1:BGC")

    ref, pre = Naive().predict(cube[VID_CHL], min_pixels=_MIN_PIXELS)
    plot_diagnostic_diagrams(ref, pre, "Naive")
    plot_diagnostic_diagrams(ref, xgb.where(pre.notnull()), "XGB1:Naive")

    ref, pre = SNaive().predict(cube[VID_CHL], min_pixels=_MIN_PIXELS)
    plot_diagnostic_diagrams(ref, pre, "sNaive")
    plot_diagnostic_diagrams(ref, xgb.where(pre.notnull()), "XGB1:sNaive")

    ref, pre = MA().predict(cube[VID_CHL], min_pixels=_MIN_PIXELS)
    plot_diagnostic_diagrams(ref, pre, "MA")
    plot_diagnostic_diagrams(ref, xgb.where(pre.notnull()), "XGB1:MA")

    ref, pre = SES().predict(cube[VID_CHL], min_pixels=_MIN_PIXELS)
    plot_diagnostic_diagrams(ref, pre, "SES")
    plot_diagnostic_diagrams(ref, xgb.where(pre.notnull()), "XGB1:SES")


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="wqf-pvr",
        description="This command produces figures for the Product "
        "Validation Report (PVR).",
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
    generate_figures(parser.parse_args())
