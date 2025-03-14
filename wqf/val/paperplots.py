#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

import os
import xarray as xr
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import seaborn as sns

# sns.set_theme(style="whitegrid")


def create_bounding_rectangle(lon_point, lat_point, offset=0.01):
    """
    Create a bounding rectangle around a single latitude and longitude point.

    Parameters:
        lon_point (float): Longitude of the central point
        lat_point (float): Latitude of the central point
        offset (float): Offset to expand the rectangle (default 0.01 degrees)

    Returns:
        dict: Bounding rectangle corners with min/max latitudes and longitudes
    """
    lat_min = lat_point - offset
    lat_max = lat_point + offset
    lon_min = lon_point - offset
    lon_max = lon_point + offset

    return {
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
    }


# Define root directory
ROOT = "."
DATA = ROOT  # os.path.join(ROOT, "Data")
FIGS = os.path.join(ROOT, "Figures")

# List all Zarr files in the directory
zarr_files = [f for f in os.listdir(DATA)]

# Dictionary to store xgb models
xgb_models = {}
cube = None

# Open and assign each dataset
for file in zarr_files:
    if not file.endswith("zarr"):
        continue
    model = file.split("_")[1]
    file_path = os.path.join(DATA, file)

    print(f"Opening {file} as {model}")
    dataset = xr.open_zarr(file_path)

    # Store cube separately
    if model == "cube":
        cube = dataset
    # Store xgb models in the dictionary
    elif model.startswith("xgb"):
        xgb_models[model] = dataset
    else:
        continue

# Select Validation year and count the number of observations
chl_2020 = cube["chl"].sel(time=slice("2020-01-01", "2020-12-31"))
pixel_count = chl_2020.count(dim=("time"))

# Select point with maximum number of observations
max_obs_location = np.unravel_index(
    pixel_count.argmax().values, pixel_count.shape
)

##############################################################################################
# Point Location Map
##############################################################################################

# Add Southern Bight ROI and points
sb_lat_point = 51.5
sb_lon_point = 2
sb_rect_coords = create_bounding_rectangle(
    sb_lon_point, sb_lat_point, offset=0.3
)

# Add German Bight ROI and points
gb_lat_point = 54.1
gb_lon_point = 8.2
gb_rect_coords = create_bounding_rectangle(
    gb_lon_point, gb_lat_point, offset=0.3
)

# Add Skagerrak ROI and points
sk_lat_point = pixel_count.lat[max_obs_location[0]].values - 0.3
sk_lon_point = pixel_count.lon[max_obs_location[1]].values
sk_rect_coords = create_bounding_rectangle(
    sk_lon_point, sk_lat_point, offset=0.3
)

text_color = "white"

mid_lat = round(pixel_count.lat.mean().item(), ndigits=1)
mid_lon = round(pixel_count.lon.mean().item(), ndigits=1)
max_lat = np.ceil(pixel_count.lat.max()).item()
min_lat = np.floor(pixel_count.lat.min()).item()

# - Plot -------------------------------------------------------------------------------------
# Create figure
fig, ax = plt.subplots(
    subplot_kw={
        "projection": ccrs.LambertConformal(
            central_longitude=mid_lon,
            central_latitude=mid_lat,
            standard_parallels=(min_lat, max_lat),
        )
    }
)

# Plot chlorophyll data
chl_plot = pixel_count.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="viridis",
    add_colorbar=True,
    robust=True,
    vmin=0.0,
    vmax=130.0,
    cbar_kwargs={"label": "number of yobservations"},
)

# Modify the colorbar label and tick colors
cbar = chl_plot.colorbar
# cbar.ax.yaxis.label.set_color(text_color)
# cbar.ax.tick_params(colors=text_color)

# Add land feature
ax.add_feature(
    cfeature.LAND, facecolor="#e8e5db", edgecolor="#e8e5db", zorder=2
)

# Add Southern Bight ROI and point
lon_start, lon_end = sb_rect_coords["lon_min"], sb_rect_coords["lon_max"]
lat_start, lat_end = sb_rect_coords["lat_min"], sb_rect_coords["lat_max"]
rect = patches.Rectangle(
    (lon_start, lat_start),
    lon_end - lon_start,
    lat_end - lat_start,
    linewidth=1,
    edgecolor="white",
    linestyle="--",
    facecolor="none",
    transform=ccrs.PlateCarree(),
    zorder=3,
)
ax.add_patch(rect)

ax.scatter(
    sb_lon_point,
    sb_lat_point,
    facecolor="white",
    edgecolor="none",
    marker="v",
    s=40,
    linewidth=1,
    transform=ccrs.PlateCarree(),
    zorder=4,
    label="Point location",
)

# Add German Bight ROI and point
lon_start, lon_end = gb_rect_coords["lon_min"], gb_rect_coords["lon_max"]
lat_start, lat_end = gb_rect_coords["lat_min"], gb_rect_coords["lat_max"]

rect = patches.Rectangle(
    (lon_start, lat_start),
    lon_end - lon_start,
    lat_end - lat_start,
    linewidth=1,
    edgecolor="white",
    linestyle="--",
    facecolor="none",
    transform=ccrs.PlateCarree(),
    zorder=3,
)

ax.add_patch(rect)

ax.scatter(
    gb_lon_point,
    gb_lat_point,
    facecolor="white",
    edgecolor="none",
    marker="v",
    s=40,
    linewidth=1,
    transform=ccrs.PlateCarree(),
    zorder=4,
    label="Point location",
)

# Add Skagerrak ROI and point
lon_start, lon_end = sk_rect_coords["lon_min"], sk_rect_coords["lon_max"]
lat_start, lat_end = sk_rect_coords["lat_min"], sk_rect_coords["lat_max"]

rect = patches.Rectangle(
    (lon_start, lat_start),
    lon_end - lon_start,
    lat_end - lat_start,
    linewidth=1,
    edgecolor="white",
    linestyle="--",
    facecolor="none",
    transform=ccrs.PlateCarree(),
    zorder=3,
)

ax.add_patch(rect)

ax.scatter(
    sk_lon_point,
    sk_lat_point,
    facecolor="white",
    edgecolor="none",
    marker="v",
    s=40,
    linewidth=1,
    transform=ccrs.PlateCarree(),
    zorder=4,
    label="Point location",
)

# Add labels for the regions
ax.text(
    sb_lon_point + 2,
    sb_lat_point + 0.5,
    "Southern Bight",
    transform=ccrs.PlateCarree(),
    ha="center",
    va="bottom",
    color=text_color,
)

ax.text(
    gb_lon_point - 2.2,
    gb_lat_point,
    "German Bight",
    transform=ccrs.PlateCarree(),
    ha="center",
    va="bottom",
    color=text_color,
)

ax.text(
    sk_lon_point - 2,
    sk_lat_point,
    "Skagerrak",
    transform=ccrs.PlateCarree(),
    ha="center",
    va="bottom",
    color=text_color,
)

# Set up gridlines
gl = ax.gridlines(
    draw_labels={"bottom": "x", "left": "y"},
    x_inline=False,
    y_inline=False,
    xlocs=[1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0],
    ylocs=[51.0, 52.5, 54.0, 55.5, 57.0],
    alpha=0.1,
)
ax.autoscale_view()
# Get current extent in PlateCarree projection
# current_extent = ax.get_extent(crs=ccrs.PlateCarree())

# lon_min_ex, lon_max_ex, lat_min_ex, lat_max_ex = current_extent
#
# longitudes = [2.5, 4.0, 5.5, 7.0, 8.5]
# offsets = [0.12, 0.09, 0.04, 0.02, 0.02]
#
# # Manually add longitude labels outside the plot
# for lon, offset in zip(longitudes, offsets):
#     ax.text(lon, lat_min_ex - offset, f"{lon}°E",
#             transform=ccrs.PlateCarree(),
#             ha='center', va='top', fontsize=12, color='gray', rotation=90)
#
# ax.set_extent([lon_min_ex + 1, lon_max_ex - 0.5, lat_min_ex, lat_max_ex],
#               crs=ccrs.PlateCarree())

# Add rectangle patch
plt.savefig("Points_locations.png", dpi=300, bbox_inches="tight")


########################################################################################################
# Skagerrak
########################################################################################################

# Add Skagerrak ROI and points
lat_point = pixel_count.lat[max_obs_location[0]].values - 0.3
lon_point = pixel_count.lon[max_obs_location[1]].values

# Extract the values for the pixel with the maximum observations over all timesteps
max_pixel_values = chl_2020.sel(
    lon=lon_point, lat=lat_point, method="nearest"
)
max_vals_time = max_pixel_values.dropna(dim="time")

# Extract values from XGB model datasets at the same pixel and time index
xgb_values = {
    model_name: xgb_models[model_name]["chl"]
    .sel(lon=lon_point, lat=lat_point, method="nearest")
    .sel(time=max_vals_time.time)
    for model_name in xgb_models.keys()
}

# Extract the 1-day forecast (darkest blue)
xgb_1d_name, xgb_1d_data = sorted(xgb_values.items(), reverse=False)[
    0
]  # The first item is 1-day forecast
xgb_1d_time = (
    xgb_1d_data.time.to_pandas()
)  # Convert XArray time to Pandas DateTimeIndex

# Define a colormap ranging from light to dark
num_models = len(xgb_values)
colors = plt.cm.Purples(
    np.linspace(0.9, 0.3, num_models)
)  # Lighter to darker shades

# Define a size range for the scatter points
min_size, max_size = 10, 30  # Adjust sizes as needed
sizes = np.linspace(max_size, min_size, num_models)

# - Scatterplot - #########################################################################################

plt.figure(figsize=(12, 5))

# Plot the original scatter in a single green color
obs_handle = plt.scatter(
    max_pixel_values.time,
    max_pixel_values,
    alpha=0.7,
    color="#009E73",
    label="observed",
    s=40,
)

# Plot the XGB model predictions with varying shades
model_labels = [
    f"{i + 1} d" for i in range(len(xgb_values))
]  # "1d", "2d", ...
scatter_handles = []

for i, ((model_name, values), new_label) in enumerate(
    zip(xgb_values.items(), model_labels)
):
    scatter = plt.scatter(
        values.time,
        values,
        alpha=0.7,
        color=colors[i],
        s=sizes[i],
        label=new_label,
    )
    scatter_handles.append(
        (int(new_label[:-1]), scatter)
    )  # Convert "1d" -> 1 for sorting

# Sort legend so "Observation" is first, then "1d", "2d", ...
scatter_handles = sorted(
    scatter_handles, key=lambda x: x[0]
)  # Sort numerically

# Connect OSPAR values to the most different forecast at the same time
for time in xgb_1d_time:
    if (
        time in max_pixel_values.time.to_pandas().values
    ):  # Ensure time exists in both
        ospar_value = max_pixel_values.sel(time=time)  # Get OSPAR value

        # Find the XGB model with the largest difference
        differences = {
            model_name: abs(values.sel(time=time) - ospar_value)
            for model_name, values in xgb_values.items()
        }
        most_diff_model = max(
            differences, key=differences.get
        )  # Model with the max difference
        forecast_value = xgb_values[most_diff_model].sel(
            time=time
        )  # Get forecast value

        # Draw a vertical dashed line
        plt.vlines(
            time,
            ospar_value,
            forecast_value,
            colors="black",
            linewidth=1,
            linestyle="--",
            alpha=0.5,
        )

plt.xlabel("time")
plt.ylabel("chlorophyll concentration (mg m⁻³)")

# Format x-axis to show only months
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))

# === Fix the legend order (Observation first, then sorted forecast days) ===
sorted_labels, sorted_handles = zip(
    *scatter_handles
)  # Extract sorted labels & handles
sorted_handles = [obs_handle] + list(
    sorted_handles
)  # Ensure "Observation" is first
sorted_labels = ["observed"] + [
    f"{d} d" for d in sorted_labels
]  # Ensure "Observation" is first

plt.legend(
    sorted_handles,
    sorted_labels,
    loc="upper center",
    # bbox_to_anchor=(0.5, -0.15),
    ncol=8,  # Keep legend compact
    frameon=False,
)
plt.xlim(np.datetime64("2020-03-02"), np.datetime64("2020-10-30"))
plt.ylim(0.0, 4.0)
plt.yticks(range(0, 5))
plt.title("Skagerrak 2020")
plt.grid(False)
plt.savefig("Skagerrak_point.pdf", dpi=300, bbox_inches="tight")
# plt.show()

# - Heatmap - #########################################################################################

# Add Skagerrak ROI
lon_start, lon_end = sk_rect_coords["lon_min"], sk_rect_coords["lon_max"]
lat_start, lat_end = sk_rect_coords["lat_min"], sk_rect_coords["lat_max"]

models_list = list(xgb_models.keys())
timeframe = slice("2020-03-01", "2020-10-31")

# Dictionary to save all the roi cubes
roi_preds = {}
# Dictionary to save all the roi cubes
flat_preds = {}
preds2d = {}

for model in models_list:
    roi_preds[model] = xgb_models[model].sel(
        lon=slice(lon_start, lon_end),
        lat=slice(lat_end, lat_start),
        time=timeframe,
    )
    flat_preds[model] = roi_preds[model].stack(pixel=("lat", "lon"))
    flat_preds[model] = flat_preds[model].transpose("pixel", "time")
    preds2d[model] = flat_preds[model]["chl"].values

roi_obs = cube["chl"].sel(
    lon=slice(lon_start, lon_end),
    lat=slice(lat_end, lat_start),
    time=timeframe,
)

flat_obs = roi_obs.stack(pixel=("lat", "lon"))

flat_obs = flat_obs.transpose("pixel", "time")
obs2d = flat_obs.values
# Convert xarray datetime DataArray to pandas Timestamps if necessary
time_dim = pd.to_datetime(cube.sel(time=timeframe).time.values)
n_rows = 1 + len(models_list)

# Plots
fig, axs = plt.subplots(
    n_rows, 1, figsize=(12, 1 * n_rows), constrained_layout=True, sharex=True
)

# Determine color limits from the obs2d
vmin, vmax = np.nanmin(obs2d), np.ceil(np.nanpercentile(obs2d, 95))

# Plotting obs2d in the first subplot with time extent
im1 = axs[0].imshow(
    obs2d,
    aspect="auto",
    cmap="viridis",
    origin="lower",
    vmin=0.0,
    vmax=4.0,
    extent=[time_dim[0], time_dim[-1], 0, 1],
)
axs[0].set_ylabel("observed")
axs[0].set_yticks([])
axs[0].grid(False)

# Loop through models and plot each in a subplot with time extent
for i, model in enumerate(models_list):
    im = axs[i + 1].imshow(
        preds2d[model],
        aspect="auto",
        cmap="viridis",
        origin="lower",
        vmin=0.0,
        vmax=4.0,
        extent=[time_dim[0], time_dim[-1], 0, 1],
    )
    axs[i + 1].set_ylabel(f"{model_labels[i]}")
    axs[i + 1].set_yticks([])
    axs[i + 1].grid(True, linewidth=1, linestyle="--", alpha=0.5)
    if i == len(models_list) - 1:  # Only the last subplot gets the xlabel
        axs[i + 1].set_xlabel("time")

axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))

# Create a colorbar that applies to all subplots
cbar = fig.colorbar(im1, ax=axs, orientation="vertical")
plt.xlim(np.datetime64("2020-03-02"), np.datetime64("2020-10-30"))
cbar.set_label("chlorophyll concentration (mg m⁻³)")  # Adding the unit label
fig.suptitle("Skagerrak 2020")
plt.savefig("Skagerrak_heatmap.png", dpi=300, bbox_inches="tight")
# plt.show()


#################################################################################################
# German Bight
#################################################################################################

# Extract the corresponding lat/lon coordinates
lat_point = 54.1
lon_point = 8.2

rect_coords = create_bounding_rectangle(lon_point, lat_point, offset=0.3)

# Extract the values for the pixel with the maximum observations over all timesteps
max_pixel_values = chl_2020.sel(
    lon=lon_point, lat=lat_point, method="nearest"
)
max_vals_time = max_pixel_values.dropna(dim="time")

# Extract values from XGB model datasets at the same pixel and time index
xgb_values = {
    model_name: xgb_models[model_name]["chl"]
    .sel(lon=lon_point, lat=lat_point, method="nearest")
    .sel(time=max_vals_time.time)
    for model_name in xgb_models.keys()
}

# Extract the 1-day forecast (darkest blue)
xgb_1d_name, xgb_1d_data = sorted(xgb_values.items(), reverse=False)[
    0
]  # The first item is 1-day forecast
xgb_1d_time = (
    xgb_1d_data.time.to_pandas()
)  # Convert XArray time to Pandas DateTimeIndex

# Define a colormap ranging from light to dark
num_models = len(xgb_values)
colors = plt.cm.Purples(
    np.linspace(0.9, 0.3, num_models)
)  # Lighter to darker shades

# Define a size range for the scatter points
min_size, max_size = 10, 30  # Adjust sizes as needed
sizes = np.linspace(max_size, min_size, num_models)

# - Scatterplot - ###################################################################################

plt.figure(figsize=(12, 5))

# Plot the original scatter in a single green color
obs_handle = plt.scatter(
    max_pixel_values.time,
    max_pixel_values,
    alpha=0.7,
    color="#009E73",
    label="observed",
    s=40,
)

# Plot the XGB model predictions with varying shades
model_labels = [
    f"{i + 1} d" for i in range(len(xgb_values))
]  # "1d", "2d", ...
scatter_handles = []

for i, ((model_name, values), new_label) in enumerate(
    zip(xgb_values.items(), model_labels)
):
    scatter = plt.scatter(
        values.time,
        values,
        alpha=0.7,
        color=colors[i],
        s=sizes[i],
        label=new_label,
    )
    scatter_handles.append(
        (int(new_label[:-1]), scatter)
    )  # Convert "1d" -> 1 for sorting

# Sort legend so "Observation" is first, then "1d", "2d", ...
scatter_handles = sorted(
    scatter_handles, key=lambda x: x[0]
)  # Sort numerically

# Connect OSPAR values to the most different forecast at the same time
for time in xgb_1d_time:
    if (
        time in max_pixel_values.time.to_pandas().values
    ):  # Ensure time exists in both
        ospar_value = max_pixel_values.sel(time=time)  # Get OSPAR value

        # Find the XGB model with the largest difference
        differences = {
            model_name: abs(values.sel(time=time) - ospar_value)
            for model_name, values in xgb_values.items()
        }
        most_diff_model = max(
            differences, key=differences.get
        )  # Model with the max difference
        forecast_value = xgb_values[most_diff_model].sel(
            time=time
        )  # Get forecast value

        # Draw a vertical dashed line
        plt.vlines(
            time,
            ospar_value,
            forecast_value,
            colors="black",
            linewidth=1,
            linestyle="--",
            alpha=0.5,
        )

plt.xlabel("")
plt.ylabel("chlorophyll concentration (mg m⁻³)")

# Format x-axis to show only months
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))

# === Fix the legend order (Observation first, then sorted forecast days) ===
sorted_labels, sorted_handles = zip(
    *scatter_handles
)  # Extract sorted labels & handles
sorted_handles = [obs_handle] + list(
    sorted_handles
)  # Ensure "Observation" is first
sorted_labels = ["observed"] + [
    f"{d} d" for d in sorted_labels
]  # Ensure "Observation" is first

plt.legend(
    sorted_handles,
    sorted_labels,
    loc="lower center",
    # bbox_to_anchor=(0.5, -0.15),
    ncol=8,  # Keep legend compact
    frameon=False,
)

plt.title("German Bight 2020")
plt.grid(False)
plt.xlim(np.datetime64("2020-03-02"), np.datetime64("2020-10-30"))
plt.ylim(0, 18)
plt.savefig("GB_point.pdf", dpi=300, bbox_inches="tight")
# plt.show()

# - Heatmap - ###################################################################################

# Extract rectangle boundaries
lon_start, lon_end = rect_coords["lon_min"], rect_coords["lon_max"]
lat_start, lat_end = rect_coords["lat_min"], rect_coords["lat_max"]

models_list = list(xgb_models.keys())
timeframe = slice("2020-03-01", "2020-12-30")

# Dictionary to save all the roi cubes
roi_preds = {}
# Dictionary to save all the roi cubes
flat_preds = {}
preds2d = {}

for model in models_list:
    roi_preds[model] = xgb_models[model].sel(
        lon=slice(lon_start, lon_end),
        lat=slice(lat_end, lat_start),
        time=timeframe,
    )
    flat_preds[model] = roi_preds[model].stack(pixel=("lat", "lon"))
    flat_preds[model] = flat_preds[model].transpose("pixel", "time")
    preds2d[model] = flat_preds[model]["chl"].values

roi_obs = cube["chl"].sel(
    lon=slice(lon_start, lon_end),
    lat=slice(lat_end, lat_start),
    time=timeframe,
)

flat_obs = roi_obs.stack(pixel=("lat", "lon"))

flat_obs = flat_obs.transpose("pixel", "time")
obs2d = flat_obs.values
# Convert xarray datetime DataArray to pandas Timestamps if necessary
time_dim = pd.to_datetime(cube.sel(time=timeframe).time.values)
n_rows = 1 + len(models_list)

# Plots
fig, axs = plt.subplots(
    n_rows, 1, figsize=(12, 1 * n_rows), constrained_layout=True, sharex=True
)

# Determine color limits from the obs2d
vmin, vmax = np.nanmin(obs2d), np.ceil(np.nanpercentile(obs2d, 95))

# Plotting obs2d in the first subplot with time extent
im1 = axs[0].imshow(
    obs2d,
    aspect="auto",
    cmap="viridis",
    origin="lower",
    vmin=0.0,
    vmax=18.0,
    extent=[time_dim[0], time_dim[-1], 0, 1],
)
axs[0].set_ylabel("observed")
axs[0].set_yticks([])
axs[0].grid(False)

# Loop through models and plot each in a subplot with time extent
for i, model in enumerate(models_list):
    im = axs[i + 1].imshow(
        preds2d[model],
        aspect="auto",
        cmap="viridis",
        origin="lower",
        vmin=0.0,
        vmax=18.0,
        extent=[time_dim[0], time_dim[-1], 0, 1],
    )
    axs[i + 1].set_ylabel(f"{model_labels[i]}")
    axs[i + 1].set_yticks([])
    axs[i + 1].grid(True, linewidth=1, linestyle="--", alpha=0.5)
    if i == len(models_list) - 1:  # Only the last subplot gets the xlabel
        axs[i + 1].set_xlabel("time")

axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))

# Create a colorbar that applies to all subplots
cbar = fig.colorbar(im1, ax=axs, orientation="vertical")
cbar.set_label("chlorophyll concetration (mg m⁻³)")  # Adding the unit label
fig.suptitle("German Bight 2020")
plt.xlim(np.datetime64("2020-03-02"), np.datetime64("2020-10-30"))
plt.savefig("GB_heatmap.png", dpi=300, bbox_inches="tight")
# plt.show()

#################################################################################################
# Southern Bight
#################################################################################################

# Extract the corresponding lat/lon coordinates
lat_point = 51.5
lon_point = 2

rect_coords = create_bounding_rectangle(lon_point, lat_point, offset=0.3)

# Extract the values for the pixel with the maximum observations over all timesteps
max_pixel_values = chl_2020.sel(
    lon=lon_point, lat=lat_point, method="nearest"
)
max_vals_time = max_pixel_values.dropna(dim="time")

# Extract values from XGB model datasets at the same pixel and time index
xgb_values = {
    model_name: xgb_models[model_name]["chl"]
    .sel(lon=lon_point, lat=lat_point, method="nearest")
    .sel(time=max_vals_time.time)
    for model_name in xgb_models.keys()
}

# Extract the 1-day forecast (darkest blue)
xgb_1d_name, xgb_1d_data = sorted(xgb_values.items(), reverse=False)[
    0
]  # The first item is 1-day forecast
xgb_1d_time = (
    xgb_1d_data.time.to_pandas()
)  # Convert XArray time to Pandas DateTimeIndex

# Define a colormap ranging from light to dark
num_models = len(xgb_values)
colors = plt.cm.Purples(
    np.linspace(0.9, 0.3, num_models)
)  # Lighter to darker shades

# Define a size range for the scatter points
min_size, max_size = 10, 30  # Adjust sizes as needed
sizes = np.linspace(max_size, min_size, num_models)

# - Scatterplot - ###################################################################################

plt.figure(figsize=(12, 5))

# Plot the original scatter in a single green color
obs_handle = plt.scatter(
    max_pixel_values.time,
    max_pixel_values,
    alpha=0.7,
    color="#009E73",
    label="observed",
    s=40,
)

# Plot the XGB model predictions with varying shades
model_labels = [
    f"{i + 1} d" for i in range(len(xgb_values))
]  # "1d", "2d", ...
scatter_handles = []

for i, ((model_name, values), new_label) in enumerate(
    zip(xgb_values.items(), model_labels)
):
    scatter = plt.scatter(
        values.time,
        values,
        alpha=0.7,
        color=colors[i],
        s=sizes[i],
        label=new_label,
    )
    scatter_handles.append(
        (int(new_label[:-1]), scatter)
    )  # Convert "1d" -> 1 for sorting

# Sort legend so "Observation" is first, then "1d", "2d", ...
scatter_handles = sorted(
    scatter_handles, key=lambda x: x[0]
)  # Sort numerically

# Connect OSPAR values to the most different forecast at the same time
for time in xgb_1d_time:
    if (
        time in max_pixel_values.time.to_pandas().values
    ):  # Ensure time exists in both
        ospar_value = max_pixel_values.sel(time=time)  # Get OSPAR value

        # Find the XGB model with the largest difference
        differences = {
            model_name: abs(values.sel(time=time) - ospar_value)
            for model_name, values in xgb_values.items()
        }
        most_diff_model = max(
            differences, key=differences.get
        )  # Model with the max difference
        forecast_value = xgb_values[most_diff_model].sel(
            time=time
        )  # Get forecast value

        # Draw a vertical dashed line
        plt.vlines(
            time,
            ospar_value,
            forecast_value,
            colors="black",
            linewidth=1,
            linestyle="--",
            alpha=0.5,
        )

plt.xlabel("time")
plt.ylabel("chlorophyll concentration (mg m⁻³)")

# Format x-axis to show only months
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))

# === Fix the legend order (Observation first, then sorted forecast days) ===
sorted_labels, sorted_handles = zip(
    *scatter_handles
)  # Extract sorted labels & handles
sorted_handles = [obs_handle] + list(
    sorted_handles
)  # Ensure "Observation" is first
sorted_labels = ["observed"] + [
    f"{d} d" for d in sorted_labels
]  # Ensure "Observation" is first

plt.legend(
    sorted_handles,
    sorted_labels,
    loc="upper center",
    # bbox_to_anchor=(0.5, -0.15),
    ncol=8,  # Keep legend compact
    frameon=False,
)
plt.ylim(0.0, 9.0)
plt.xlim(np.datetime64("2020-03-02"), np.datetime64("2020-10-30"))
plt.title("Southern Bight 2020")
plt.grid(False)
plt.savefig("SB_point.pdf", dpi=300, bbox_inches="tight")
# plt.show()

# - Heatmap - ###################################################################################

# Extract rectangle boundaries
lon_start, lon_end = rect_coords["lon_min"], rect_coords["lon_max"]
lat_start, lat_end = rect_coords["lat_min"], rect_coords["lat_max"]

models_list = list(xgb_models.keys())
timeframe = slice("2020-03-01", "2020-12-31")

# Dictionary to save all the roi cubes
roi_preds = {}
# Dictionary to save all the roi cubes
flat_preds = {}
preds2d = {}

for model in models_list:
    roi_preds[model] = xgb_models[model].sel(
        lon=slice(lon_start, lon_end),
        lat=slice(lat_end, lat_start),
        time=timeframe,
    )
    flat_preds[model] = roi_preds[model].stack(pixel=("lat", "lon"))
    flat_preds[model] = flat_preds[model].transpose("pixel", "time")
    preds2d[model] = flat_preds[model]["chl"].values

roi_obs = cube["chl"].sel(
    lon=slice(lon_start, lon_end),
    lat=slice(lat_end, lat_start),
    time=timeframe,
)

flat_obs = roi_obs.stack(pixel=("lat", "lon"))

flat_obs = flat_obs.transpose("pixel", "time")
obs2d = flat_obs.values
# Convert xarray datetime DataArray to pandas Timestamps if necessary
time_dim = pd.to_datetime(cube.sel(time=timeframe).time.values)
n_rows = 1 + len(models_list)

# Plots
fig, axs = plt.subplots(
    n_rows, 1, figsize=(12, 1 * n_rows), constrained_layout=True, sharex=True
)

# Determine color limits from the obs2d
vmin, vmax = np.nanmin(obs2d), np.ceil(np.nanpercentile(obs2d, 95))

# Plotting obs2d in the first subplot with time extent
im1 = axs[0].imshow(
    obs2d,
    aspect="auto",
    cmap="viridis",
    origin="lower",
    vmin=0.0,
    vmax=8.0,
    extent=[time_dim[0], time_dim[-1], 0, 1],
)
axs[0].set_ylabel("observed")
axs[0].set_yticks([])
axs[0].grid(False)

# Loop through models and plot each in a subplot with time extent
for i, model in enumerate(models_list):
    im = axs[i + 1].imshow(
        preds2d[model],
        aspect="auto",
        cmap="viridis",
        origin="lower",
        vmin=0.0,
        vmax=8.0,
        extent=[time_dim[0], time_dim[-1], 0, 1],
    )
    axs[i + 1].set_ylabel(f"{model_labels[i]}")
    axs[i + 1].set_yticks([])
    axs[i + 1].grid(True, linewidth=1, linestyle="--", alpha=0.5)
    if i == len(models_list) - 1:  # Only the last subplot gets the xlabel
        axs[i + 1].set_xlabel("time")

axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))

# Create a colorbar that applies to all subplots
cbar = fig.colorbar(im1, ax=axs, orientation="vertical")
plt.xlim(np.datetime64("2020-03-02"), np.datetime64("2020-10-30"))
cbar.set_label("chlorophyll concentration (mg m⁻³)")  # Adding the unit label
fig.suptitle("Southern Bight 2020")
plt.savefig("SB_heatmap.png", dpi=300, bbox_inches="tight")
# plt.show()
