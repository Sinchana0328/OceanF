import xarray as xr
import numpy as np
from pathlib import Path


# ============================================================
# 1. File paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Currents"
    / "Currents.nc"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Currents"
    / "Currents_processed.nc"
)


# ============================================================
# 2. Open raw dataset
# ============================================================

print("Opening raw Currents dataset...")

ds = xr.open_dataset(INPUT_FILE)

print(ds)


# ============================================================
# 3. Select current components
# ============================================================

uo = ds["uo"].astype("float32")
vo = ds["vo"].astype("float32")


# ============================================================
# 4. Remove singleton depth dimension
# ============================================================

print("Removing singleton depth dimension...")

uo = uo.squeeze("depth", drop=True)
vo = vo.squeeze("depth", drop=True)


# ============================================================
# 5. Preserve metadata
# ============================================================

uo.attrs = ds["uo"].attrs.copy()
vo.attrs = ds["vo"].attrs.copy()


# ============================================================
# 6. Remove invalid grid_mapping reference
# ============================================================

uo.attrs.pop("grid_mapping", None)
vo.attrs.pop("grid_mapping", None)


# ============================================================
# 7. Add descriptions
# ============================================================

uo.attrs["description"] = (
    "Eastward surface ocean current velocity at 0 m depth"
)

vo.attrs["description"] = (
    "Northward surface ocean current velocity at 0 m depth"
)


# ============================================================
# 8. Create processed dataset
# ============================================================

ds_processed = xr.Dataset(
    {
        "uo": uo,
        "vo": vo,
    },
    coords={
        "time": ds["time"],
        "latitude": ds["latitude"],
        "longitude": ds["longitude"],
    },
    attrs=ds.attrs.copy(),
)


# ============================================================
# 9. Add actual time coverage
# ============================================================

ds_processed.attrs["time_coverage_start"] = (
    str(ds["time"].values[0])[:19]
)

ds_processed.attrs["time_coverage_end"] = (
    str(ds["time"].values[-1])[:19]
)


# ============================================================
# 10. Create output directory
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 11. Save processed dataset
# ============================================================

print("Saving processed Currents...")

encoding = {
    "uo": {
        "dtype": "float32",
        "zlib": True,
        "complevel": 4,
        "_FillValue": np.float32(-9999.0),
    },
    "vo": {
        "dtype": "float32",
        "zlib": True,
        "complevel": 4,
        "_FillValue": np.float32(-9999.0),
    },
}


ds_processed.to_netcdf(
    OUTPUT_FILE,
    encoding=encoding,
)


# ============================================================
# 12. Close files
# ============================================================

ds.close()
ds_processed.close()


print()
print("==========================================")
print("Currents preprocessing completed successfully")
print("==========================================")
print(f"Output file: {OUTPUT_FILE}")