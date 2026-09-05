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

print("\nTemperature variable:")
print(ds["thetao"])

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