<video controls>
<source src="https://github.com/user-attachments/assets/c3677da7-d674-4ba3-bb6f-694701ef1087" type="video/mp4">
</video>

# Forecasting Water Quality from Space (FC-WQ)

> Ever tried. Ever failed. No matter. Try again. Fail again. Fail better. (Samuel Becket)

> Attention is all you need. ([Ashish Vaswani *et al.*](https://dl.acm.org/doi/10.5555/3295222.3295349))

> [XGBoost](https://xgboost.readthedocs.io/) is all you need. (Bojan Tunguz)

## Synopsis

This repository includes the forecast processor authored for the **Forecasting
Water Quality from Space** study proposed by and granted to Brockmann Consult
GmbH in reply to ESA's Invitation to Tender for Future  EO-1 EO Science for
Society Permanently Open Call for Proposals, ESA Ref. ESA AO/1-10468/20/I-FvO.

The objective of the study is to provide time series forecasts of chlorophyll
concentration. Predicted chlorophyll concentration is derived from EO-based
retrievals of chlorophyll concentration and several covariates, which are
routinely provided by reanalysis and forecast services based on biological,
chemical, physical, and meteorological models.

[![Package](https://github.com/bcdev/fcwq-wqf/actions/workflows/python-package.yml/badge.svg)](https://github.com/bcdev/fcwq-wqf/actions/workflows/python-package.yml)
[![CodeQL Advanced](https://github.com/bcdev/fcwq-wqf/actions/workflows/codeql.yml/badge.svg)](https://github.com/bcdev/fcwq-wqf/actions/workflows/codeql.yml)
[![codecov](https://codecov.io/github/bcdev/fcwq-wqf/graph/badge.svg?token=UNdi5jfQML)](https://codecov.io/github/bcdev/fcwq-wqf)

## Installing and testing

Read [INSTALL.md](INSTALL.md) for detailed instructions on installation
and testing.

## Operations Manual

### Operational principles

The processor is coded in Python and requires an environment described
in [INSTALL.md](INSTALL.md). The [wqf](wqf) directory includes the `wqf`
Python package, which includes the water quality forecast (WQF) processor.
The processor is invoked from the command line. Typing

    wqf --help

will print a detailed usage message to the screen:

    usage: wqf [-h] [--aws] [--chunk-size-lat CHUNK_SIZE_LAT]
               [--chunk-size-lon CHUNK_SIZE_LON] [--depth-level {0.0,3.0,10.0}]
               [--engine-reader {h5netcdf,netcdf4,zarr}]
               [--engine-writer {h5netcdf,netcdf4,zarr}]
               [--gaussian-filter {0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0}]
               [--horizon {1,2,3,4,5,6,7}]
               [--log-level {debug,info,warning,error,off}]
               [--mode {multithreading,synchronous}] [--model MODEL]
               [--nthread {1,2,3,4,5,6,7,8}] [--workers {1,2,3,4,5,6,7,8}]
               [--prof PROF] [--progress] [--no-progress] [--stack-traces]
               [--no-stack-traces] [--test] [--no-test] [--tmpdir TMPDIR] [-v]
               source_id target_id
    
    This scientific processor conducts time series forecasts of water quality.
    
    positional arguments:
      source_id             the identifier of the source dataset.
      target_id             the identifier of the target dataset.
    
    options:
      -h, --help            show this help message and exit
      --aws                 read from and write to AWS team store (default:
                            False)
      --chunk-size-lat CHUNK_SIZE_LAT
                            specify the chunk size along the latitudinal
                            dimension for reading and computing data arrays. A
                            value of `-1` refers to full latitudinal chunk size
                            and a value of `0` refers to the chunk size used in
                            the source file. (default: None)
      --chunk-size-lon CHUNK_SIZE_LON
                            specify the chunk size along the longitudinal
                            dimension for reading and computing data arrays. A
                            value of `-1` refers to full longitudinal chunk size
                            and a value of `0` refers to the chunk size used in
                            the source file. (default: None)
      --depth-level {0.0,3.0,10.0}
                            specify the depth level (m) to consider, if the
                            source dataset exhibits a depth coordinate.
                            (default: None)
      --engine-reader {h5netcdf,netcdf4,zarr}
                            specify the engine used to read the source product
                            file. (default: None)
      --engine-writer {h5netcdf,netcdf4,zarr}
                            specify the engine used to write the target product
                            file. (default: None)
      --gaussian-filter GAUSSIAN_FILTER
                            specify the full width at half maximum (pixels) of
                            a lateral Gaussian filter applied to the forecast.
                            If not specified no filter is applied. (default:
                            None)
      --horizon {1,2,3,4,5,6,7}
                            specify the forecast horizon (days). (default: None)
      --log-level {debug,info,warning,error,off}
                            specify the log level. (default: None)
      --mode {multithreading,synchronous}
                            specify the operating mode. In multithreading mode a
                            multithreading scheduler is used. In synchronous
                            mode a single-thread scheduler is used. (default:
                            None)
      --model MODEL         specify the forecast model. Either a model name or a
                            file path to a custom model definition. Valid model
                            names are {default, ns-central, ns-coastal, ns-
                            natural}. If not specified, the default model is
                            used. (default: None)
      --nthread {1,2,3,4,5,6,7,8}
                            specify the number of threads used by the forecast
                            model. If not set, the number of threads is
                            determined by the system. (default: None)
      --workers {1,2,3,4,5,6,7,8}
                            specify the number of workers used in multithreading
                            mode. If not set, the number of workers is
                            determined by the system. (default: None)
      --prof PROF           specify the path to the file containing profiling
                            information. If set, the processor will operate in
                            synchronous mode and will generate profiling
                            information. If not set explicitly, no profiling is
                            conducted. (default: None)
      --progress            enable progress bar display. (default: False)
      --no-progress         disable progress bar display. (default: True)
      --stack-traces        enable Python stack traces. (default: False)
      --no-stack-traces     disable Python stack traces. (default: True)
      --test                enable forecast test mode for verification.
                            (default: False)
      --no-test             disable forecast test mode. (default: True)
      --tmpdir TMPDIR       specify the path to the temporary directory.
                            (default: None)
      -v, --version         show program's version number and exit
    
    Copyright (c) Brockmann Consult GmbH, 2024. License: MIT

### Normal operations

To invoke the processor from the terminal, for instance, type 

    wqf --horizon 3 \
        --chunk-size-lat -1 \
        --chunk-size-lon -1 \
        --engine-reader h5netcdf \
        --engine-writer h5netcdf \
        in.nc fc.nc

which normally will log information to the terminal, e.g.,

    2024-11-25T10:49:16.848000Z <node> wqf 2025.1.0 [6667] [I] starting running processor
    2024-11-25T10:49:16.848000Z <node> wqf 2025.1.0 [6667] [I] config: chunk_size_lat = -1
    2024-11-25T10:49:16.848000Z <node> wqf 2025.1.0 [6667] [I] config: chunk_size_lon = -1
    2024-11-25T10:49:16.848000Z <node> wqf 2025.1.0 [6667] [I] config: depth_level = 3.0
    2024-11-25T10:49:16.848000Z <node> wqf 2025.1.0 [6667] [I] config: engine_reader = h5netcdf
    2024-11-25T10:49:16.848000Z <node> wqf 2025.1.0 [6667] [I] config: engine_writer = h5netcdf
    2024-11-25T10:49:16.848000Z <node> wqf 2025.1.0 [6667] [I] config: horizon = 3
    2024-11-25T10:49:16.848000Z <node> wqf 2025.1.0 [6667] [I] config: log_level = info
    2024-11-25T10:49:16.848000Z <node> wqf 2025.1.0 [6667] [I] config: mode = multithreading
    2024-11-25T10:49:16.848000Z <node> wqf 2025.1.0 [6667] [I] config: model = default
    2024-11-25T10:49:16.848000Z <node> wqf 2025.1.0 [6667] [I] config: nthread = None
    2024-11-25T10:49:16.848000Z <node> wqf 2025.1.0 [6667] [I] config: processor_name = wqf
    2024-11-25T10:49:16.848000Z <node> wqf 2025.1.0 [6667] [I] config: processor_version = 2025.1.0
    2024-11-25T10:49:16.848000Z <node> wqf 2025.1.0 [6667] [I] config: prof = None
    2024-11-25T10:49:16.848000Z <node> wqf 2025.1.0 [6667] [I] config: progress = False
    2024-11-25T10:49:16.848000Z <node> wqf 2025.1.0 [6667] [I] config: source_id = in.nc
    2024-11-25T10:49:16.848000Z <node> wqf 2025.1.0 [6667] [I] config: stack_traces = True
    2024-11-25T10:49:16.848000Z <node> wqf 2025.1.0 [6667] [I] config: target_id = fc.nc
    2024-11-25T10:49:16.848000Z <node> wqf 2025.1.0 [6667] [I] config: test = False
    2024-11-25T10:49:16.848000Z <node> wqf 2025.1.0 [6667] [I] config: tmpdir = .
    2024-11-25T10:49:16.848000Z <node> wqf 2025.1.0 [6667] [I] config: workers = None
    2024-11-25T10:49:17.138000Z <node> wqf 2025.1.0 [6667] [I] starting creating processing graph
    2024-11-25T10:49:17.161000Z <node> wqf 2025.1.0 [6667] [I] finished creating processing graph
    2024-11-25T10:49:17.161000Z <node> wqf 2025.1.0 [6667] [I] starting writing target dataset: fc.nc
    2024-11-25T10:49:18.877000Z <node> wqf 2025.1.0 [6667] [I] finished writing target dataset
    2024-11-25T10:49:18.877000Z <node> wqf 2025.1.0 [6667] [I] starting closing datasets
    2024-11-25T10:49:18.878000Z <node> wqf 2025.1.0 [6667] [I] finished closing datasets
    2024-11-25T10:49:18.879000Z <node> wqf 2025.1.0 [6667] [I] finished running processor
    2024-11-25T10:49:18.879000Z <node> wqf 2025.1.0 [6667] [I] elapsed time (seconds):    2.031

and eventually produce a forecast output dataset. Normally, the processor
will terminate with an exit code of `0`. 

### Error conditions

The processor terminates on the first occurrence of an error. The exit code
of the processor is `0` if the processor completed without errors, and nonzero
otherwise. Warning and error messages are sent to the standard error stream. 

### Recovery operations

There are no recovery operations. Since the target file could be corrupted in
case of abnormally interrupted or aborted processing, it shall be deleted unless
a log message confirms that the target file was written and closed.
