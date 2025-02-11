#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This module defines dataset, dimension and variable identifiers.
"""

BGCM_ID = "FCWQ_bgcm_2016-2020_v1.0.zarr"
"""
The BGCM cube identifier.

The BGCM cube contains chlorophyll concentration obtained from
Copernicus Service reanalysis using a biogeochemical model. The
BGCM cube is used for comparison only.
"""

CUBE_ID = "FCWQ_cube_2016-2020_v3.0.zarr"
"""
The Data Cube identifier.

The Data Cube contains chlorophyll concentration obtained from the OSPAR
Commission and reanalysis data obtained from Copernicus Service. The Data
Cube is used for training and validation of the XGB forecast model.
"""

XGBM_ID = "FCWQ_xgbH_2016-2020_v1.0.zarr"
"""
The XGBM cube identifier.

XGBM cubes were produced by running the WQF processor in test mode. The
capital letter `H` is a placeholder for the forecast horizon.
"""

DID_TIM = "time"
"""The time dimension identifier."""

DID_LAT = "lat"
"""The latitude dimension identifier."""

DID_LON = "lon"
"""The longitude dimension identifier."""

DID_DEP = "depth"
"""The depth dimension identifier."""

VID_TIM = "time"
"""The time variable identifier."""

VID_LAT = "lat"
"""The latitude variable identifier."""

VID_LON = "lon"
"""The longitude variable identifier."""

VID_CHL = "chl"
"""The chlorophyll concentration (mg m-3) variable identifier."""

VID_DEP = "deptho"
"""The sea floor depth (m) variable identifier."""

VID_NO3 = "no3"
"""
The concentration of nitrates (mmol m-3) variable identifier.

The infinite values of this variable determine where chlorophyll
concentration forecasts are eventually nullified.
"""

VID_RAN = "random"
"""The random number variable identifier (used for training only)."""

VID_WGT = "weight"
"""The sample weight variable identifier (used for training only)."""
