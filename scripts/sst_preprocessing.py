import xarray as xr
import numpy as np
from pathlib import Path


# ============================================================
# 1. File paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "SST" / "SST.nc"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "SST" / "SST_processed.nc"


# ============================================================
# 2. Open raw SST dataset
# ============================================================

print("Opening raw SST dataset...")

ds = xr.open_dataset(INPUT_FILE)

print(ds)


# ============================================================
# 3. Select SST variable
# ============================================================

sst = ds["analysed_sst"]


# ============================================================
# 4. Convert Kelvin → Celsius
# ============================================================

print("Converting SST from Kelvin to Celsius...")

sst_celsius = sst - 273.15

sst_celsius = sst_celsius.astype("float32")


# ============================================================
# 5. Update SST metadata
# ============================================================

sst_celsius.attrs = sst.attrs.copy()

sst_celsius.attrs["units"] = "degree_Celsius"
sst_celsius.attrs["long_name"] = "analysed sea surface temperature"
sst_celsius.attrs["description"] = (
    "Sea surface temperature converted from Kelvin to Celsius"
)


# ============================================================
# 6. Create processed dataset
# ============================================================

ds_processed = xr.Dataset(
    {
        "sst": sst_celsius
    },
    coords={
        "time": ds["time"],
        "latitude": ds["latitude"],
        "longitude": ds["longitude"],
    },
    attrs=ds.attrs.copy(),
)


# ============================================================
# 7. Correct time metadata
# ============================================================

ds_processed.attrs["time_coverage_start"] = (
    str(ds["time"].values[0])[:19]
)

ds_processed.attrs["time_coverage_end"] = (
    str(ds["time"].values[-1])[:19]
)


# ============================================================
# 8. Remove stale packed-data metadata
# ============================================================

ds_processed["sst"].attrs.pop("valid_min", None)
ds_processed["sst"].attrs.pop("valid_max", None)


# ============================================================
# 9. Create output directory
# ============================================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 10. Save processed SST
# ============================================================

print("Saving processed SST...")

encoding = {
    "sst": {
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
print("SST preprocessing completed successfully")
print("==========================================")
print(f"Output file: {OUTPUT_FILE}")