#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This module produces plots for the Test (Site) Data Report (TDR).
"""
import warnings
from argparse import ArgumentDefaultsHelpFormatter
from argparse import ArgumentParser

import numpy as np
import xarray as xr
from matplotlib import colors as plc

from wqf.interface.constants import DID_LAT
from wqf.interface.constants import DID_LON
from wqf.interface.constants import DID_TIM
from wqf.readerfactory import ReaderFactory
from wqf.val.plots import DensityPlot
from wqf.val.plots import HistogramPlot
from wqf.val.plots import ScenePlot

warnings.filterwarnings(
    "ignore",
)


def chlorophyll_quantiles():
    """
    There are virtually no spots of apparently high chlorophyll
    concentration in open waters, which could be artifacts of
    undetected clouds.
    """
    ScenePlot().plot(
        chl_q_lo,
        title="quantile q = 0.005",
        fn="fig01",
        cbar_label=r"chlorophyll concentration (mg m$^{-3}$)",
        norm=plc.LogNorm(),
        xlocs=(1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0),
        vmin=1.0,
        vmax=100.0,
    ).clear()
    ScenePlot().plot(
        chl_q_hi,
        title="quantile q = 0.995",
        fn="fig02",
        cbar_label=r"chlorophyll concentration (mg m$^{-3}$)",
        norm=plc.LogNorm(),
        xlocs=(1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0),
        vmin=1.0,
        vmax=100.0,
    ).clear()


def chlorophyll_mean_and_std():
    """
    Spots of relatively high variability of chlorophyll concentration in
    open water reveal undetected clouds.

    Contamination of chlorophyll values is rare, however. We could get
    rid of cloud contamination, if we excluded the upper percentile of
    chlorophyll concentration values.
    """
    ScenePlot().plot(
        cube.chl.mean(DID_TIM),
        title="mean",
        cbar_label=r"chlorophyll concentration (mg m$^{-3}$)",
        norm=plc.LogNorm(),
        xlocs=(1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0),
        vmin=1.0,
        vmax=100.0,
    ).clear()
    ScenePlot().plot(
        cube.chl.std(DID_TIM),
        title="standard deviation",
        cbar_label=r"chlorophyll concentration (mg m$^{-3}$)",
        norm=plc.LogNorm(),
        xlocs=(1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0),
        vmin=1.0,
        vmax=100.0,
    ).clear()
    ScenePlot().plot(
        xr.where(cube.chl < chl_q_hi, cube.chl, np.nan).mean(DID_TIM),
        title="mean",
        fn="fig03",
        cbar_label=r"chlorophyll concentration (mg m$^{-3}$)",
        norm=plc.LogNorm(),
        xlocs=(1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0),
        vmin=1.0,
        vmax=100.0,
    ).clear()
    ScenePlot().plot(
        xr.where(cube.chl < chl_q_hi, cube.chl, np.nan).std(DID_TIM),
        title="standard deviation",
        fn="fig04",
        cbar_label=r"chlorophyll concentration (mg m$^{-3}$)",
        norm=plc.LogNorm(),
        xlocs=(1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0),
        vmin=1.0,
        vmax=100.0,
    ).clear()


def chlorophyll_variability():
    """
    The variability of the (spatial) mean chlorophyll concentration within
    the annual cycle illustrates blooms in spring and fall.

    The figure considers coastal waters only, which are our main interest.
    """
    DensityPlot().plot(
        (
            cube.doy,
            xr.where(cube.deptho < 30.0, cube.chl, np.nan).mean(
                [DID_LAT, DID_LON]
            ),
        ),
        xlabel="day of year",
        ylabel=r"chlorophyll concentration (mg m$^{-3}$)",
        title="mean CHL vs day of year",
        fn="fig05",
        bins=(50, 50),
        cbar_label="number count",
        density=False,
        hist_range=((40, 310), (0.0, 10)),
        vmin=0.0,
        vmax=6.0,
    ).clear()


def number_of_chlorophyll_observations_from_space():
    """
    There is less than one observation from space per day due to cloud
    coverage, tidal effects, and revisit cycles, which is no surprise
    but stresses the importance of using a machine learning model that
    can learn from training data with gaps.

    In fact, we have more gaps than observations, which in our opinion
    prohibits the use of gap filling strategies. We do not want to train
    our model with mainly artificial data.

    The histogram illustrates that high values of chlorophyll concentration
    occur very rarely. Chlorophyll concentrations higher than 20 mg m-3 are
    already exceptional. To achieve a more balanced training, we thought on
    means to emphasize high concentrations.

    Just for your interest: the statistical distribution of values of
    chlorophyll concentration can be well described by an exponential
    distribution with a mean value of about 1.0 mg m-3.
    """
    chl_count = cube.chl.count(dim="time") / cube.chl.shape[0]
    chl_count = xr.where(chl_count == 0, np.nan, 1.0 * chl_count)

    ScenePlot().plot(
        chl_count,
        title="number of observations",
        fn="fig06",
        cbar_label=r"number of observations (day$^{-1}$)",
        xlocs=(1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0),
        vmin=0.1,
        vmax=0.5,
    ).clear()
    HistogramPlot().plot(
        cube.chl,
        xlabel=r"chlorophyll concentration (mg m$^{-3}$)",
        ylabel="number count",
        ylim=(100, 100000000),
        title="number distribution",
        fn="fig07",
        bins=25,
        log=True,
        hist_range=(0.0, 100.0),
    ).clear()


def depth_of_sea_floor():
    """
    The mean chlorophyll concentration is quite correlated with the depth
    of the sea floor.

    Simply put, chlorophyll concentration decreases with increasing depth.
    This is a statistical correlation and not necessarily a causal
    relation. Algae flourish better with nutrients provided by river
    run-off and mixing due to tidal effects, which both are stronger in
    shallow water near the coast.

    The variability of chlorophyll concentrations correlates with depth,
    too. Seasonal effects and spontaneous events concern shallow water
    mainly.
    """
    ScenePlot().plot(
        cube.deptho,
        title="sea floor depth",
        fn="fig08",
        cbar_label="depth (m)",
        xlocs=(1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0),
    ).clear()
    DensityPlot().plot(
        (cube.deptho, cube.chl.mean(DID_TIM)),
        xlabel="depth (m)",
        ylabel=r"chlorophyll concentration (mg m$^{-3}$)",
        title="mean CHL vs sea floor depth",
        fn="fig09",
        bins=(50, 50),
        cbar_label="number count",
        density=False,
        hist_range=((0.0, 50.0), (0.0, 50.0)),
        norm=plc.SymLogNorm(1.0, vmin=0.0, vmax=2000.0),
    ).clear()
    DensityPlot().plot(
        (cube.deptho, cube.chl.std(DID_TIM)),
        xlabel="depth (m)",
        ylabel=r"chlorophyll concentration (mg m$^{-3}$)",
        title="standard deviation CHL vs sea floor depth",
        fn="fig10",
        bins=(50, 50),
        cbar_label="number count",
        density=False,
        hist_range=((0.0, 50.0), (0.0, 50.0)),
        norm=plc.SymLogNorm(1.0, vmin=0.0, vmax=2000.0),
    ).clear()


def examples_of_statistical_correlations():
    """
    Correlation plots reveal interesting relationships to tease the brain.

    In principle, every relationship can be explained (though we cannot
    explain all of them). Gradient boosting machine learning models (GBM)
    facilitate feature importance analysis as part of their training. We
    rely on this analysis instead of investigating every possible
    correlation separately.
    """
    DensityPlot().plot(
        (cube.mdt, cube.chl.mean(DID_TIM)),
        xlabel="mean dynamic topography (m)",
        ylabel=r"chlorophyll concentration (mg m$^{-3}$)",
        title="mean CHL vs mean dynamic topography",
        fn="fig11",
        bins=(50, 50),
        cbar_label="number count",
        density=False,
        hist_range=((-0.53, -0.27), (0.0, 50.0)),
        norm=plc.SymLogNorm(1.0, vmin=0.0, vmax=2000.0),
    ).clear()
    DensityPlot().plot(
        (cube.sst.mean(DID_TIM), cube.chl.mean(DID_TIM)),
        xlabel="sea surface temperature (K)",
        ylabel=r"chlorophyll concentration (mg m$^{-3}$)",
        title="mean CHL vs mean SST",
        fn="fig12",
        bins=(50, 50),
        cbar_label="number count",
        density=False,
        hist_range=((284.0, 289.0), (0.0, 50.0)),
        norm=plc.SymLogNorm(1.0, vmin=0.0, vmax=2000.0),
    ).clear()
    DensityPlot().plot(
        (cube.sst.std(DID_TIM), cube.chl.mean(DID_TIM)),
        xlabel="sea surface temperature (K)",
        ylabel=r"chlorophyll concentration (mg m$^{-3}$)",
        title="mean CHL vs standard deviation SST",
        fn="fig13",
        bins=(50, 50),
        cbar_label="number count",
        density=False,
        hist_range=((2.0, 7.0), (0.0, 50.0)),
        norm=plc.SymLogNorm(1.0, vmin=0.0, vmax=2000.0),
    ).clear()
    DensityPlot().plot(
        (cube.so.mean(DID_TIM), cube.chl.mean(DID_TIM)),
        xlabel="surface salinity (10-3)",
        ylabel=r"chlorophyll concentration (mg m$^{-3}$)",
        title="mean CHL vs mean surface salinity",
        fn="fig14",
        bins=(50, 50),
        cbar_label="number count",
        density=False,
        hist_range=((0.0, 50.0), (0.0, 50.0)),
        norm=plc.SymLogNorm(1.0, vmin=0.0, vmax=2000.0),
    ).clear()
    DensityPlot().plot(
        (cube.mlotst.mean(DID_TIM), cube.chl.mean(DID_TIM)),
        xlabel="mixed layer thickness (m)",
        ylabel=r"chlorophyll concentration (mg m$^{-3}$)",
        title="mean CHL vs mean mixed layer thickness",
        fn="fig15",
        bins=(50, 50),
        cbar_label="number count",
        density=False,
        hist_range=((0.0, 50.0), (0.0, 50.0)),
        norm=plc.SymLogNorm(1.0, vmin=0.0, vmax=2000.0),
    ).clear()
    DensityPlot().plot(
        (cube.mlotst.mean(DID_TIM), cube.deptho),
        xlabel="sea floor depth (m)",
        ylabel="mixed layer thickness (m)",
        title="mean mixed layer thickness vs sea floor depth",
        fn="fig16",
        bins=(50, 50),
        cbar_label="number count",
        density=False,
        hist_range=((0.0, 200.0), (0.0, 50.0)),
        norm=plc.SymLogNorm(1.0, vmin=0.0, vmax=2000.0),
    ).clear()


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="wqf-tdr",
        description="This command produces figures for the Test (Site) Data "
        "Report (TDR).",
        epilog="Copyright (c) Brockmann Consult GmbH, 2024. License: MIT",
        exit_on_error=True,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("cube_id", help="the data cube identifier")
    parser.add_argument(
        "--aws",
        help="read from AWS team store",
        action="store_true",
        default=False,
        required=False,
        dest="aws",
    )

    args = parser.parse_args()
    reader = ReaderFactory.create_reader(args.aws)
    cube = reader.read(args.cube_id, depth_level=3.0, unify=False)
    chl_q_lo = cube.chl.quantile(0.005, dim=DID_TIM).compute()
    chl_q_hi = cube.chl.quantile(0.995, dim=DID_TIM).compute()

    chlorophyll_quantiles()

    chlorophyll_mean_and_std()

    chlorophyll_variability()

    number_of_chlorophyll_observations_from_space()

    depth_of_sea_floor()

    examples_of_statistical_correlations()
