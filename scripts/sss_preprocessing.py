import xarray as xr
import numpy as np
from pathlib import Path


# ============================================================
# 1. File paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "SSS" / "SSS.nc"
OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "SSS" / "SSS_processed.nc"
)


# ============================================================
# 2. Open raw SSS dataset
# ============================================================

print("Opening raw SSS dataset...")

ds = xr.open_dataset(INPUT_FILE)

print(ds)


# ============================================================
# 3. Select SSS variable
# ============================================================

sss = ds["sos"]


# ============================================================
# 4. Remove singleton depth dimension
# ============================================================

print("Removing singleton depth dimension...")

sss = sss.squeeze("depth", drop=True)


# ============================================================
# 5. Convert to float32
# ============================================================

sss = sss.astype("float32")


# ============================================================
# 6. Update metadata
# ============================================================

sss.attrs = ds["sos"].attrs.copy()

sss.attrs["description"] = (
    "Sea surface salinity at 0 m depth"
)


# ============================================================
# 7. Create processed dataset
# ============================================================

ds_processed = xr.Dataset(
    {
        "sss": sss
    },
    coords={
        "time": ds["time"],
        "latitude": ds["latitude"],
        "longitude": ds["longitude"],
    },
    attrs=ds.attrs.copy(),
)


# ============================================================
# 8. Correct time coverage metadata
# ============================================================

ds_processed.attrs["time_coverage_start"] = (
    str(ds["time"].values[0])[:19]
)

ds_processed.attrs["time_coverage_end"] = (
    str(ds["time"].values[-1])[:19]
)


# ============================================================
# 9. Create output directory
# ============================================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 10. Save processed SSS
# ============================================================

print("Saving processed SSS...")

encoding = {
    "sss": {
        "dtype": "float32",
        "zlib": True,
        "complevel": 4,
        "_FillValue": np.float32(-9999.0),
    }
}

ds_processed.to_netcdf(
    OUTPUT_FILE,
    encoding=encoding,
)


# ============================================================
# 11. Close datasets
# ============================================================

ds.close()
ds_processed.close()


print()
print("==========================================")
print("SSS preprocessing completed successfully")
print("==========================================")
print(f"Output file: {OUTPUT_FILE}")