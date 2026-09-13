"""
InVEST SDR (Sediment Delivery Ratio) model runner — Bhasan Char
================================================================
Reuses the same DEM, R-factor, K-factor, LULC, and AOI inputs prepared for
the RUSLE pipeline, so no new data collection is needed.

Run inside the 'invest' conda environment:
    conda env create -f invest_environment.yml
    conda activate invest
    python invest_sdr.py

Edit the CONFIG section below to point at your actual data/ folder paths.

Output (per year, in invest_outputs/<year>/):
    rkls.tif           - gross potential erosion (R x K x LS), directly
                          comparable to your RUSLE soil_loss_A.tif before
                          applying C and P
    usle.tif            - full USLE/RUSLE equation (R x K x LS x C x P) -
                          the file to cross-check against RUSLE's
                          soil_loss_A_<year>_clipped.tif
    sed_export.tif      - sediment actually reaching the watershed outlet
    sed_retention.tif   - sediment trapped before reaching the outlet
    sed_deposition.tif  - where sediment is deposited
    watershed_results_sdr.shp - AOI-level totals (export, retention, etc.)
"""

import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from natcap.invest.sdr import sdr

# ---------------------------------------------------------------------
# CONFIG — edit these paths to match your data/ folder
# ---------------------------------------------------------------------
DATA_DIR = "data"
OUT_DIR = "invest_outputs"
CLEAN_DIR = "invest_clean_inputs"  # standardized, reprojected copies of the raw rasters

# Target projected CRS for InVEST (SDR needs a METRIC CRS, not lat/lon
# degrees, or its internal slope-length/flow calculations are wrong by a
# large, consistent scale factor). UTM Zone 46N covers Bhasan Char's
# longitude (~91.4E) and all of central/eastern Bangladesh.
TARGET_CRS = "EPSG:32646"

DEM_PATH = os.path.join(DATA_DIR, "bhasanchar_dem_copernicus.tif")
R_FACTOR_PATH = os.path.join(DATA_DIR, "bhasanchar_r_factor.tif")
K_FACTOR_PATH = os.path.join("rusle_outputs", "k_factor.tif")  # reuse RUSLE's derived K-factor
AOI_PATH = os.path.join(DATA_DIR, "Bhasanchar.shp")
BIOPHYSICAL_TABLE_PATH = "biophysical_table.csv"

LULC_PATHS = {
    "2015": os.path.join(DATA_DIR, "clip2015.tif"),
    "2020": os.path.join(DATA_DIR, "clip2020.tif"),
    "2025": os.path.join(DATA_DIR, "clip2025.tif"),
}

# SDR calibration parameters — sensible defaults; tune later if needed
THRESHOLD_FLOW_ACCUMULATION = 100  # smaller = more of the flat island counted as "stream"
K_PARAM = 2                        # SDR calibration constant (InVEST default)
IC_0_PARAM = 0.5                   # connectivity index calibration constant (InVEST default)
SDR_MAX = 0.8                      # InVEST default
L_MAX = 122                        # max slope length in meters, InVEST default

CLEAN_NODATA = -9999.0


def clean_raster(src_path, dst_path, is_int=False, resampling=None):
    """
    Re-writes a raster reprojected into TARGET_CRS (a metric UTM CRS),
    with a well-formed, explicit NoData value and a standard dtype.

    Two problems this fixes at once:
    1. Some source GeoTIFFs (e.g. the DEM) carry a malformed NoData tag
       (like '-32768.0'), which crashed InVEST with a cryptic
       'NoneType has no attribute lower' error.
    2. All of the source rasters are in geographic coordinates (EPSG:4326,
       degrees). InVEST's SDR model needs a projected/metric CRS — feeding
       it degree-based pixel spacing silently produces RKLS/USLE values off
       by many orders of magnitude (it treats each ~0.00027-degree pixel as
       if it were 0.00027 *meters* apart internally). Reprojecting to UTM
       46N (EPSG:32646) here fixes that at the source.
    """
    if resampling is None:
        resampling = Resampling.nearest if is_int else Resampling.bilinear

    with rasterio.open(src_path) as src:
        arr = src.read(1).astype("float64")
        old_nodata = src.nodata
        if old_nodata is not None:
            arr[arr == old_nodata] = np.nan
        arr[np.isnan(arr)] = CLEAN_NODATA

        transform, width, height = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(
            crs=TARGET_CRS, transform=transform, width=width, height=height,
            nodata=CLEAN_NODATA, compress="lzw",
        )

        if is_int:
            dst_arr = np.full((height, width), CLEAN_NODATA, dtype="float64")
        else:
            dst_arr = np.full((height, width), CLEAN_NODATA, dtype="float64")

        reproject(
            source=arr,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=CLEAN_NODATA,
            dst_transform=transform,
            dst_crs=TARGET_CRS,
            dst_nodata=CLEAN_NODATA,
            resampling=resampling,
        )

    if is_int:
        dst_arr = np.round(dst_arr).astype("int32")
        profile.update(dtype="int32")
    else:
        dst_arr = dst_arr.astype("float32")
        profile.update(dtype="float32")

    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(dst_arr, 1)
    print(f"  cleaned + reprojected ({TARGET_CRS}) {src_path} -> {dst_path}")


def prepare_clean_inputs():
    Path(CLEAN_DIR).mkdir(parents=True, exist_ok=True)
    clean_paths = {}

    clean_paths["dem"] = os.path.join(CLEAN_DIR, "dem.tif")
    clean_raster(DEM_PATH, clean_paths["dem"])

    clean_paths["r_factor"] = os.path.join(CLEAN_DIR, "r_factor.tif")
    clean_raster(R_FACTOR_PATH, clean_paths["r_factor"])

    clean_paths["k_factor"] = os.path.join(CLEAN_DIR, "k_factor.tif")
    clean_raster(K_FACTOR_PATH, clean_paths["k_factor"])

    clean_paths["lulc"] = {}
    for year, path in LULC_PATHS.items():
        out_path = os.path.join(CLEAN_DIR, f"lulc_{year}.tif")
        clean_raster(path, out_path, is_int=True)
        clean_paths["lulc"][year] = out_path

    return clean_paths


def ensure_watershed_id(aoi_path):
    """
    InVEST's SDR model needs the watershed vector to have a unique integer
    ID field, and it should match the same projected CRS used for the
    rasters. Rather than trying to fix dtype issues in the original
    shapefile's attribute columns (which can trip up fiona's schema
    inference in some geopandas versions), we just build a fresh, minimal
    GeoDataFrame with only the geometry and a ws_id column, reprojected.
    """
    gdf = gpd.read_file(aoi_path)
    gdf = gdf.to_crs(TARGET_CRS)

    minimal_gdf = gpd.GeoDataFrame(
        {"ws_id": range(1, len(gdf) + 1)},
        geometry=gdf.geometry.values,
        crs=TARGET_CRS,
    )
    fixed_path = os.path.join(DATA_DIR, "Bhasanchar_ws_id.shp")
    minimal_gdf.to_file(fixed_path)
    print(f"[setup] wrote {fixed_path} with a ws_id field, reprojected to {TARGET_CRS}")
    return fixed_path


def run_year(year, dem_path, r_factor_path, k_factor_path, lulc_path, watershed_path):
    workspace = os.path.join(OUT_DIR, year)
    Path(workspace).mkdir(parents=True, exist_ok=True)
    print(f"\n=== Running InVEST SDR for {year} ===")

    args = {
        "workspace_dir": workspace,
        "results_suffix": "",
        "dem_path": dem_path,
        "erosivity_path": r_factor_path,
        "erodibility_path": k_factor_path,
        "lulc_path": lulc_path,
        "watersheds_path": watershed_path,
        "biophysical_table_path": BIOPHYSICAL_TABLE_PATH,
        "threshold_flow_accumulation": THRESHOLD_FLOW_ACCUMULATION,
        "k_param": K_PARAM,
        "ic_0_param": IC_0_PARAM,
        "sdr_max": SDR_MAX,
        "l_max": L_MAX,
        # NOTE: no "drainage_path" key at all — this is an OPTIONAL input.
        # Passing it as "" causes an internal AttributeError in some InVEST
        # versions (it expects the key to be absent, not empty, when you
        # don't have a known drainage/stream layer).
    }

    sdr.execute(args)
    print(f"[done] {year} -> {workspace}/")


def main():
    watershed_path = ensure_watershed_id(AOI_PATH)

    print("=== Standardizing input rasters (fixing NoData metadata) ===")
    clean_paths = prepare_clean_inputs()

    for year, lulc_path in clean_paths["lulc"].items():
        run_year(year, clean_paths["dem"], clean_paths["r_factor"],
                  clean_paths["k_factor"], lulc_path, watershed_path)

    print("\nAll years complete. Key outputs per year, in invest_outputs/<year>/:")
    print("  rkls.tif            - gross erosion (compare pattern to RUSLE's LS x R x K)")
    print("  usle.tif             - full RUSLE equation (compare directly to soil_loss_A_<year>_clipped.tif)")
    print("  sed_export.tif       - sediment reaching the outlet")
    print("  watershed_results_sdr.shp - AOI-level summary totals")


if __name__ == "__main__":
    main()
