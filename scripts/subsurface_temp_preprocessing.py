import xarray as xr
import numpy as np
from pathlib import Path
import time


# ============================================================
# CONFIGURATION
# ============================================================

RAW_FILE = Path(
    r"C:\OceanF\data\raw\SubsurfaceTemp\SubsurfaceTemp.nc"
)

OUTPUT_DIR = Path(
    r"C:\OceanF\data\processed\SubsurfaceTemp"
)

OUTPUT_FILE = OUTPUT_DIR / "SubsurfaceTemp_processed.nc"


# OceanF target depths (metres)
TARGET_DEPTHS = np.array([
    0,
    5,
    10,
    20,
    30,
    50,
    75,
    100,
    125,
    150,
    200,
    300,
    500,
    700,
    1000
], dtype=np.float32)


# OceanF target spatial resolution
TARGET_RESOLUTION = 0.25

MIN_LAT = 5.0
MAX_LAT = 30.0

MIN_LON = 45.0
MAX_LON = 105.0


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 70)
print("OceanF - Subsurface Temperature Preprocessing")
print("=" * 70)

print(f"Raw file    : {RAW_FILE}")
print(f"Output file : {OUTPUT_FILE}")
print(f"Target depths: {TARGET_DEPTHS}")

# ============================================================
# OPEN RAW GLORYS DATASET
# ============================================================

print("\n========== OPENING RAW DATASET ==========")

start_time = time.time()

ds = xr.open_dataset(
    RAW_FILE,
    chunks={
        "time": 1,
        "depth": 36,
        "latitude": 301,
        "longitude": 721
    }
)

print(ds)

thetao = ds["thetao"]

print("\nTemperature variable:")
print(thetao)

print(
    f"\nDataset opened in "
    f"{time.time() - start_time:.2f} seconds"
)
# ============================================================
# VALIDATE RAW DATASET
# ============================================================

print("\n========== VALIDATING RAW DATASET ==========")

expected_dimensions = {
    "time": 184,
    "depth": 36,
    "latitude": 301,
    "longitude": 721
}

for dim, expected_size in expected_dimensions.items():

    actual_size = ds.sizes[dim]

    print(
        f"{dim:10s}: "
        f"{actual_size} "
        f"(expected {expected_size})"
    )

    if actual_size != expected_size:
        raise ValueError(
            f"Unexpected {dim} dimension size: "
            f"{actual_size}"
        )


if thetao.dims != (
    "time",
    "depth",
    "latitude",
    "longitude"
):
    raise ValueError(
        f"Unexpected thetao dimensions: {thetao.dims}"
    )


if thetao.attrs.get("units") != "degrees_C":
    raise ValueError(
        f"Unexpected temperature units: "
        f"{thetao.attrs.get('units')}"
    )


if thetao.attrs.get("standard_name") != \
        "sea_water_potential_temperature":

    raise ValueError(
        "thetao is not identified as "
        "sea_water_potential_temperature."
    )


print("\nRaw dataset validation PASSED.")

print("\n========== CREATING TARGET GRID ==========")

target_lat = np.arange(
    MIN_LAT,
    MAX_LAT + TARGET_RESOLUTION / 2,
    TARGET_RESOLUTION,
    dtype=np.float32
)

target_lon = np.arange(
    MIN_LON,
    MAX_LON + TARGET_RESOLUTION / 2,
    TARGET_RESOLUTION,
    dtype=np.float32
)

print(f"Target latitude points : {len(target_lat)}")
print(f"Target longitude points: {len(target_lon)}")

print(f"Latitude range : {target_lat[0]} → {target_lat[-1]}")
print(f"Longitude range: {target_lon[0]} → {target_lon[-1]}")

if len(target_lat) != 101:
    raise ValueError(
        f"Expected 101 latitude points, got {len(target_lat)}"
    )

if len(target_lon) != 241:
    raise ValueError(
        f"Expected 241 longitude points, got {len(target_lon)}"
    )

print("\nTarget grid creation PASSED.")

print("\n========== CHECKING GRID ALIGNMENT ==========")

native_lat = ds.latitude.values
native_lon = ds.longitude.values


def find_exact_indices(native_coordinates, target_coordinates, coordinate_name):
    indices = []

    for target in target_coordinates:
        differences = np.abs(native_coordinates - target)

        index = int(np.argmin(differences))
        difference = differences[index]

        if difference > 1e-5:
            raise ValueError(
                f"{coordinate_name} target {target} does not align "
                f"with native grid. Difference = {difference}"
            )

        indices.append(index)

    return np.array(indices, dtype=np.int64)


lat_indices = find_exact_indices(
    native_lat,
    target_lat,
    "Latitude"
)

lon_indices = find_exact_indices(
    native_lon,
    target_lon,
    "Longitude"
)


print("\nFirst latitude indices:")
print(lat_indices[:15])

print("\nFirst longitude indices:")
print(lon_indices[:15])

print("\nLatitude index differences:")
print(np.diff(lat_indices[:15]))

print("\nLongitude index differences:")
print(np.diff(lon_indices[:15]))


if not np.all(np.diff(lat_indices) == 3):
    raise ValueError(
        "Latitude indices do not follow expected 3-point spacing."
    )

if not np.all(np.diff(lon_indices) == 3):
    raise ValueError(
        "Longitude indices do not follow expected 3-point spacing."
    )


print("\nGrid alignment PASSED.")

print("\n========== EXTRACTING 0.25° GRID ==========")

start_time = time.time()

thetao_target_grid = thetao.isel(
    latitude=lat_indices,
    longitude=lon_indices
)

print(thetao_target_grid)

print("\nExtracted shape:", thetao_target_grid.shape)

expected_extracted_shape = (184, 36, 101, 241)

if thetao_target_grid.shape != expected_extracted_shape:
    raise ValueError(
        f"Unexpected extracted shape: {thetao_target_grid.shape}"
    )

print(
    f"\nGrid extraction completed in "
    f"{time.time() - start_time:.2f} seconds"
)

print("\n0.25° grid extraction PASSED.")

print("\n========== VERIFYING TARGET COORDINATES ==========")

actual_lat = thetao_target_grid.latitude.values
actual_lon = thetao_target_grid.longitude.values

if not np.allclose(
    actual_lat,
    target_lat,
    atol=1e-5
):
    raise ValueError(
        "Extracted latitude coordinates do not match target grid."
    )

if not np.allclose(
    actual_lon,
    target_lon,
    atol=1e-5
):
    raise ValueError(
        "Extracted longitude coordinates do not match target grid."
    )

print(f"Latitude : {actual_lat[0]} → {actual_lat[-1]}")
print(f"Longitude: {actual_lon[0]} → {actual_lon[-1]}")

print(f"Latitude points : {len(actual_lat)}")
print(f"Longitude points: {len(actual_lon)}")

print("\nTarget coordinates VERIFIED.")

print("\n========== PREPARING VERTICAL INTERPOLATION ==========")

native_depths = thetao_target_grid.depth.values

print("Native depth levels:")
print(native_depths)

print("\nTarget depths:")
print(TARGET_DEPTHS)

# -------------------------------------------------------
# Validate that all target depths except 0 m are inside
# the native GLORYS depth range.
# -------------------------------------------------------

interpolation_depths = TARGET_DEPTHS[TARGET_DEPTHS > 0]

if interpolation_depths.min() < native_depths.min():
    raise ValueError(
        "A target depth is shallower than the native GLORYS depth range."
    )

if interpolation_depths.max() > native_depths.max():
    raise ValueError(
        "A target depth is deeper than the native GLORYS depth range."
    )

print("\nVertical interpolation range check PASSED.")

print(
    f"Native depth range : "
    f"{native_depths.min():.3f} → {native_depths.max():.3f} m"
)

print(
    f"Interpolation range: "
    f"{interpolation_depths.min():.1f} → "
    f"{interpolation_depths.max():.1f} m"
)

print(
    f"\n0 m will use the shallowest native level: "
    f"{native_depths[0]:.6f} m"
)


print("\n========== INTERPOLATING TO TARGET DEPTHS ==========")

start_time = time.time()

# -------------------------------------------------------
# Target depths greater than 0 m
# 0 m will be handled separately using the shallowest
# native GLORYS level (~0.494 m).
# -------------------------------------------------------

positive_target_depths = TARGET_DEPTHS[TARGET_DEPTHS > 0]

print("\nInterpolating to:")
print(positive_target_depths)

# -------------------------------------------------------
# Perform true numerical linear interpolation along
# the native GLORYS depth coordinate.
# -------------------------------------------------------

thetao_interpolated = thetao_target_grid.interp(
    depth=xr.DataArray(
        positive_target_depths,
        dims="target_depth"
    ),
    method="linear"
)

print("\nInterpolated DataArray:")
print(thetao_interpolated)

print("\nInterpolated dimensions:")
print(thetao_interpolated.dims)

print("\nInterpolated shape:")
print(thetao_interpolated.shape)

expected_interpolated_shape = (
    184,
    14,
    101,
    241
)

if thetao_interpolated.shape != expected_interpolated_shape:
    raise ValueError(
        f"Unexpected interpolated shape: "
        f"{thetao_interpolated.shape}"
    )

print(
    f"\nVertical interpolation setup completed in "
    f"{time.time() - start_time:.2f} seconds"
)

print("\nVertical interpolation PASSED.")

print("\n========== ADDING 0 m SURFACE PROXY ==========")

start_time = time.time()

# -------------------------------------------------------
# Extract the shallowest native GLORYS level.
#
# Native shallowest depth:
# ~0.494 m
#
# This is used as the OceanF 0 m surface proxy.
# No extrapolation is performed above the native grid.
# -------------------------------------------------------

thetao_surface = thetao_target_grid.isel(
    depth=0
)

print("Surface proxy:")
print(thetao_surface)

print(
    f"\nNative surface depth: "
    f"{float(native_depths[0]):.6f} m"
)

print(
    f"Surface proxy shape: "
    f"{thetao_surface.shape}"
)

expected_surface_shape = (
    184,
    101,
    241
)

if thetao_surface.shape != expected_surface_shape:
    raise ValueError(
        f"Unexpected surface proxy shape: "
        f"{thetao_surface.shape}"
    )

print("\nSurface proxy extraction PASSED.")

# ============================================================
# STAGE 6 — COMBINE ALL TARGET DEPTHS
# ============================================================

print("\n========== COMBINING TARGET DEPTHS ==========")

start_time = time.time()

# ------------------------------------------------------------
# Prepare interpolated data
# ------------------------------------------------------------
# The interpolation result has:
#   dimensions = (time, target_depth, latitude, longitude)
#
# Convert target_depth into the final depth dimension.
thetao_interpolated = thetao_interpolated.rename(
    {"target_depth": "depth"}
)

# Explicitly assign the target depth values.
thetao_interpolated = thetao_interpolated.assign_coords(
    depth=positive_target_depths
)

print("Interpolated depths:")
print(thetao_interpolated.depth.values)

# ------------------------------------------------------------
# Prepare 0 m surface proxy
# ------------------------------------------------------------
thetao_surface = thetao_surface.expand_dims(
    depth=[0.0]
)

print("\nSurface proxy after adding depth dimension:")
print(thetao_surface)

# ------------------------------------------------------------
# Combine 0 m + interpolated depths
# ------------------------------------------------------------
thetao_final = xr.concat(
    [
        thetao_surface,
        thetao_interpolated
    ],
    dim="depth"
)

# ------------------------------------------------------------
# IMPORTANT:
# Force final OceanF dimension order
# ------------------------------------------------------------
thetao_final = thetao_final.transpose(
    "time",
    "depth",
    "latitude",
    "longitude"
)

print("\nFinal DataArray:")
print(thetao_final)

print("\nFinal dimensions:")
print(thetao_final.dims)

print("\nFinal shape:")
print(thetao_final.shape)

# ------------------------------------------------------------
# Validate final shape
# ------------------------------------------------------------
expected_final_shape = (
    184,
    15,
    101,
    241
)

if thetao_final.shape != expected_final_shape:
    raise ValueError(
        f"Unexpected final shape: "
        f"{thetao_final.shape}. "
        f"Expected: {expected_final_shape}"
    )

# ------------------------------------------------------------
# Validate final depth coordinates
# ------------------------------------------------------------
expected_depths = TARGET_DEPTHS.astype(np.float32)

actual_depths = thetao_final.depth.values

if not np.allclose(
    actual_depths,
    expected_depths,
    atol=1e-5
):
    raise ValueError(
        f"Final depth coordinates do not match target depths.\n"
        f"Expected: {expected_depths}\n"
        f"Actual:   {actual_depths}"
    )

print("\nFinal depth coordinates:")
print(actual_depths)

print(
    f"\nDepth combination completed in "
    f"{time.time() - start_time:.2f} seconds"
)

print("\nTARGET DEPTH COMBINATION PASSED.")

# ============================================================
# STAGE 7 — PREPARE FINAL DATASET
# ============================================================

print("\n========== PREPARING FINAL DATASET ==========")

start_time = time.time()

# ------------------------------------------------------------
# Convert temperature to float32
# ------------------------------------------------------------
# This reduces storage substantially while retaining
# appropriate precision for ocean temperature reconstruction.
thetao_final = thetao_final.astype(np.float32)

# ------------------------------------------------------------
# Set variable name
# ------------------------------------------------------------
thetao_final.name = "thetao"

# ------------------------------------------------------------
# Clean variable metadata
# ------------------------------------------------------------
thetao_final.attrs = {
    "long_name": "Subsurface sea water potential temperature",
    "standard_name": "sea_water_potential_temperature",
    "units": "degrees_C",
    "description": (
        "Daily subsurface ocean potential temperature reconstructed "
        "onto standardized target depths. Values at 5–1000 m are "
        "obtained by linear interpolation between native GLORYS "
        "vertical levels. The 0 m level uses the shallowest native "
        "GLORYS level at approximately 0.494 m as a surface proxy."
    ),
    "vertical_interpolation": "Linear interpolation",
    "surface_proxy_depth_m": float(native_depths[0]),
    "spatial_method": (
        "Exact native-grid point selection at 0.25 degree resolution"
    ),
}

# ------------------------------------------------------------
# Create final dataset
# ------------------------------------------------------------
ds_final = thetao_final.to_dataset()

# ------------------------------------------------------------
# Clean coordinate metadata
# ------------------------------------------------------------
ds_final["time"].attrs = {
    "standard_name": "time"
}

ds_final["depth"].attrs = {
    "standard_name": "depth",
    "long_name": "Target depth",
    "units": "m",
    "positive": "down",
}

ds_final["latitude"].attrs = {
    "standard_name": "latitude",
    "long_name": "Latitude",
    "units": "degrees_north",
}

ds_final["longitude"].attrs = {
    "standard_name": "longitude",
    "long_name": "Longitude",
    "units": "degrees_east",
}

# ------------------------------------------------------------
# Dataset metadata
# ------------------------------------------------------------
ds_final.attrs = {
    "title": (
        "OceanF North Indian Ocean Daily Subsurface "
        "Temperature Dataset"
    ),
    "project": "OceanF",
    "description": (
        "Daily subsurface ocean potential temperature over the "
        "North Indian Ocean derived from GLORYS12V1 reanalysis."
    ),
    "source": "MERCATOR GLORYS12V1",
    "source_variable": "thetao",
    "domain": "North Indian Ocean",
    "latitude_range": "5N to 30N",
    "longitude_range": "45E to 105E",
    "spatial_resolution": "0.25 degree",
    "temporal_resolution": "Daily",
    "vertical_levels": 15,
    "target_depths_m": (
        "0, 5, 10, 20, 30, 50, 75, 100, "
        "125, 150, 200, 300, 500, 700, 1000"
    ),
    "vertical_method": (
        "Linear interpolation from native GLORYS depth levels"
    ),
    "surface_method": (
        "Shallowest native level (~0.494 m) used as 0 m proxy"
    ),
    "spatial_method": (
        "Exact native-grid point selection; no spatial averaging"
    ),
    "missing_data_policy": (
        "Original GLORYS land and shallow-bathymetry NaNs preserved"
    ),
    "processing_stage": "Processed",
    "created_for": "OceanF satellite embedding and deep learning framework",
}

# ------------------------------------------------------------
# Validate final dataset structure
# ------------------------------------------------------------
print("\nFinal dataset:")
print(ds_final)

print("\nFinal variable:")
print(ds_final["thetao"])

print("\nFinal dtype:")
print(ds_final["thetao"].dtype)

print("\nFinal dimensions:")
print(ds_final["thetao"].dims)

print("\nFinal shape:")
print(ds_final["thetao"].shape)

# ------------------------------------------------------------
# Structural validation
# ------------------------------------------------------------
if ds_final["thetao"].dtype != np.float32:
    raise ValueError(
        f"Unexpected dtype: {ds_final['thetao'].dtype}"
    )

if ds_final["thetao"].dims != (
    "time",
    "depth",
    "latitude",
    "longitude"
):
    raise ValueError(
        f"Unexpected dimension order: "
        f"{ds_final['thetao'].dims}"
    )

if ds_final["thetao"].shape != (
    184,
    15,
    101,
    241
):
    raise ValueError(
        f"Unexpected final shape: "
        f"{ds_final['thetao'].shape}"
    )

print(
    f"\nDataset preparation completed in "
    f"{time.time() - start_time:.2f} seconds"
)

print("\nFINAL DATASET PREPARATION PASSED.")

# ============================================================
# WRITE FINAL NETCDF
# ============================================================

print("\n" + "=" * 70)
print("WRITING FINAL NETCDF")
print("=" * 70)

print(f"Output file:")
print(OUTPUT_FILE)

encoding = {
    "thetao": {
        "dtype": "float32",
        "zlib": True,
        "complevel": 1,
        "shuffle": True,
        "_FillValue": np.float32(np.nan),
    }
}

print("\nCompression:")
print("  dtype     : float32")
print("  zlib      : True")
print("  level     : 1")
print("  shuffle   : True")

print("\nWriting dataset...")
print("Please DO NOT press Ctrl+C.")
print("This may take several minutes.\n")

ds_final.to_netcdf(
    OUTPUT_FILE,
    mode="w",
    format="NETCDF4",
    encoding=encoding
)

print("\n" + "=" * 70)
print("NETCDF WRITING COMPLETED")
print("=" * 70)

print(f"\nOutput file successfully created:")
print(OUTPUT_FILE)

if OUTPUT_FILE.exists():

    file_size_gb = (
        OUTPUT_FILE.stat().st_size / (1024 ** 3)
    )

    print(
        f"File size: {file_size_gb:.3f} GB"
    )

print("\nSUBSURFACE TEMPERATURE PREPROCESSING COMPLETED.")