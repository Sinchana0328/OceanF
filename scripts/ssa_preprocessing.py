import xarray as xr
import numpy as np
from pathlib import Path


# ============================================================
# 1. File paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "SSA" / "SSA.nc"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "SSA"
    / "SSA_processed.nc"
)


# ============================================================
# 2. Open raw SSA dataset
# ============================================================

print("Opening raw SSA dataset...")

ds = xr.open_dataset(INPUT_FILE)

print(ds)


# ============================================================
# 3. Select SSA variables
# ============================================================

sla = ds["sla"].astype("float32")
err_ugosa = ds["err_ugosa"].astype("float32")
vgos = ds["vgos"].astype("float32")


# ============================================================
# 4. Preserve variable metadata
# ============================================================

sla.attrs = ds["sla"].attrs.copy()
err_ugosa.attrs = ds["err_ugosa"].attrs.copy()
vgos.attrs = ds["vgos"].attrs.copy()


# ============================================================
# 5. Remove references to variables that are not retained
# ============================================================

# The original SLA metadata refers to "err_sla",
# but that variable is not present in this dataset.
sla.attrs.pop("ancillary_variables", None)

# The original variables refer to a "crs" grid mapping,
# but no CRS variable is present in the dataset we are retaining.
sla.attrs.pop("grid_mapping", None)
err_ugosa.attrs.pop("grid_mapping", None)
vgos.attrs.pop("grid_mapping", None)


# ============================================================
# 6. Add useful descriptions
# ============================================================

sla.attrs["description"] = (
    "Sea level anomaly referenced to the [1993, 2012] mean sea surface"
)

err_ugosa.attrs["description"] = (
    "Formal mapping error on zonal geostrophic velocity anomaly"
)

vgos.attrs["description"] = (
    "Absolute northward geostrophic velocity"
)


# ============================================================
# 7. Create processed dataset
# ============================================================

ds_processed = xr.Dataset(
    {
        "sla": sla,
        "err_ugosa": err_ugosa,
        "vgos": vgos,
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
# 10. Save processed SSA
# ============================================================

print("Saving processed SSA...")

encoding = {
    "sla": {
        "dtype": "float32",
        "zlib": True,
        "complevel": 4,
        "_FillValue": np.float32(-9999.0),
    },
    "err_ugosa": {
        "dtype": "float32",
        "zlib": True,
        "complevel": 4,
        "_FillValue": np.float32(-9999.0),
    },
    "vgos": {
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
# 11. Close datasets
# ============================================================

ds.close()
ds_processed.close()


print()
print("==========================================")
print("SSA preprocessing completed successfully")
print("==========================================")
print(f"Output file: {OUTPUT_FILE}")