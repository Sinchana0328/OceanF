import xarray as xr
import numpy as np
from pathlib import Path

# ============================================================
# OceanF — Final Subsurface Temperature QC
# ============================================================

PROCESSED_FILE = Path(
    r"C:\OceanF\data\processed\SubsurfaceTemp\SubsurfaceTemp_processed.nc"
)

EXPECTED_DEPTHS = np.array([
    0, 5, 10, 20, 30, 50, 75, 100,
    125, 150, 200, 300, 500, 700, 1000
], dtype=np.float32)

print("=" * 70)
print("OceanF - Final Subsurface Temperature QC")
print("=" * 70)

print(f"\nFile:")
print(PROCESSED_FILE)

if not PROCESSED_FILE.exists():
    raise FileNotFoundError(
        f"Processed file does not exist:\n{PROCESSED_FILE}"
    )

file_size_gb = PROCESSED_FILE.stat().st_size / (1024 ** 3)

print(f"File size: {file_size_gb:.3f} GB")

# ------------------------------------------------------------
# Open processed dataset
# ------------------------------------------------------------

print("\n========== OPENING PROCESSED DATASET ==========")

ds = xr.open_dataset(PROCESSED_FILE)

print(ds)

# ------------------------------------------------------------
# Basic structure
# ------------------------------------------------------------

print("\n========== STRUCTURE CHECK ==========")

expected_dims = {
    "time": 184,
    "depth": 15,
    "latitude": 101,
    "longitude": 241,
}

for dim, expected_size in expected_dims.items():

    actual_size = ds.sizes[dim]

    print(
        f"{dim:10s}: "
        f"{actual_size} (expected {expected_size})"
    )

    if actual_size != expected_size:
        raise ValueError(
            f"Unexpected {dim} size: "
            f"{actual_size}"
        )

# ------------------------------------------------------------
# Variable check
# ------------------------------------------------------------

print("\n========== VARIABLE CHECK ==========")

if "thetao" not in ds:
    raise ValueError("thetao variable not found.")

thetao = ds["thetao"]

print("Variable : thetao")
print(f"Dtype    : {thetao.dtype}")
print(f"Dims     : {thetao.dims}")
print(f"Shape    : {thetao.shape}")

if thetao.dtype != np.float32:
    raise ValueError(
        f"thetao dtype is {thetao.dtype}, "
        f"expected float32"
    )

expected_dims_order = (
    "time",
    "depth",
    "latitude",
    "longitude",
)

if thetao.dims != expected_dims_order:
    raise ValueError(
        f"Unexpected dimension order: {thetao.dims}"
    )

# ------------------------------------------------------------
# Depth check
# ------------------------------------------------------------

print("\n========== DEPTH CHECK ==========")

actual_depths = ds["depth"].values.astype(np.float32)

print("Actual depths:")
print(actual_depths)

if not np.allclose(
    actual_depths,
    EXPECTED_DEPTHS,
    atol=1e-5
):
    raise ValueError(
        "Depth coordinates do not match expected target depths."
    )

print("Depth coordinates PASSED.")

# ------------------------------------------------------------
# Spatial coordinate check
# ------------------------------------------------------------

print("\n========== SPATIAL CHECK ==========")

lat = ds["latitude"].values
lon = ds["longitude"].values

print(
    f"Latitude : {lat.min()} → {lat.max()} "
    f"({len(lat)} points)"
)

print(
    f"Longitude: {lon.min()} → {lon.max()} "
    f"({len(lon)} points)"
)

if not np.isclose(lat.min(), 5.0):
    raise ValueError("Latitude minimum incorrect.")

if not np.isclose(lat.max(), 30.0):
    raise ValueError("Latitude maximum incorrect.")

if not np.isclose(lon.min(), 45.0):
    raise ValueError("Longitude minimum incorrect.")

if not np.isclose(lon.max(), 105.0):
    raise ValueError("Longitude maximum incorrect.")

if not np.allclose(
    np.diff(lat),
    0.25,
    atol=1e-5
):
    raise ValueError(
        "Latitude spacing is not 0.25 degree."
    )

if not np.allclose(
    np.diff(lon),
    0.25,
    atol=1e-5
):
    raise ValueError(
        "Longitude spacing is not 0.25 degree."
    )

print("Spatial coordinates PASSED.")

# ------------------------------------------------------------
# Time check
# ------------------------------------------------------------

print("\n========== TIME CHECK ==========")

time = ds["time"].values

print(f"Start: {time[0]}")
print(f"End  : {time[-1]}")
print(f"Count: {len(time)}")

time_diffs = np.diff(time).astype("timedelta64[D]")

if not np.all(time_diffs == np.timedelta64(1, "D")):
    raise ValueError(
        "Time coordinate is not daily."
    )

print("Daily time spacing PASSED.")

# ------------------------------------------------------------
# Targeted data read
# ------------------------------------------------------------

print("\n========== TARGETED DATA CHECK ==========")

# Read only a very small subset.
sample = thetao.isel(
    time=[0, -1],
    depth=[0, 6, 10, 14],
    latitude=50,
    longitude=100
).load()

print("Sample values:")
print(sample.values)

if not np.all(
    np.isnan(sample.values) |
    np.isfinite(sample.values)
):
    raise ValueError(
        "Sample contains invalid numerical values."
    )

finite_sample = sample.values[
    np.isfinite(sample.values)
]

if finite_sample.size > 0:

    print(
        f"\nSample temperature range: "
        f"{finite_sample.min():.3f} → "
        f"{finite_sample.max():.3f} °C"
    )

# ------------------------------------------------------------
# Missing-data check
# ------------------------------------------------------------

print("\n========== MISSING-DATA CHECK ==========")

# Use a single time slice to avoid a full 269 MB scan.
sample_slice = thetao.isel(time=0).load()

missing_count = int(
    np.isnan(sample_slice.values).sum()
)

total_count = sample_slice.size

missing_fraction = (
    missing_count / total_count
)

print(f"Time slice checked: {str(time[0])[:10]}")
print(f"Missing values    : {missing_count}")
print(f"Total values      : {total_count}")
print(f"Missing fraction  : {missing_fraction:.4%}")

print(
    "\nNote: GLORYS land/shallow-bathymetry NaNs "
    "are expected and preserved."
)

# ------------------------------------------------------------
# Metadata check
# ------------------------------------------------------------

print("\n========== METADATA CHECK ==========")

required_attrs = [
    "title",
    "project",
    "source",
    "domain",
    "spatial_resolution",
    "temporal_resolution",
    "vertical_method",
    "surface_method",
    "spatial_method",
    "missing_data_policy",
]

for attr in required_attrs:

    if attr not in ds.attrs:
        raise ValueError(
            f"Missing dataset attribute: {attr}"
        )

    print(f"{attr}: {ds.attrs[attr]}")

required_variable_attrs = [
    "long_name",
    "standard_name",
    "units",
    "vertical_interpolation",
    "surface_proxy_depth_m",
    "spatial_method",
]

for attr in required_variable_attrs:

    if attr not in thetao.attrs:
        raise ValueError(
            f"Missing thetao attribute: {attr}"
        )

print("\nVariable metadata PASSED.")

# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL QC PASSED")
print("=" * 70)

print("\nOceanF subsurface temperature dataset is ready.")

print("\nFinal dataset:")
print(
    "time × depth × latitude × longitude = "
    f"{thetao.shape}"
)

print(f"\nFile size: {file_size_gb:.3f} GB")

ds.close()