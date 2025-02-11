#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This module provides netCDF utility functions.

The module make use of system programs for reading and writing
netCDF files usually included with the `netcdf-bin` package.
"""

from os import system
from pathlib import Path


def ncdump(nc_file: str | Path) -> Path:
    """Dumps a netCDF file into a CDL file."""
    if isinstance(nc_file, str):
        nc_file = Path(nc_file)
    cdl_file_name = nc_file.name.replace(".nc", ".cdl")
    cdl_file = nc_file.parent.joinpath(cdl_file_name)
    system("ncdump -c -p 4,8 -s {0} > {1}".format(nc_file, cdl_file))
    return cdl_file


def ncgen(cdl_file: str | Path) -> Path:
    """Generates a netCDF file from a CDL file."""
    if isinstance(cdl_file, str):
        cdl_file = Path(cdl_file)
    nc_file_name = cdl_file.name.replace(".cdl", ".nc")
    nc_file = cdl_file.parent.joinpath(nc_file_name)
    system(f"ncgen -4 -o {nc_file} {cdl_file}")
    return nc_file
