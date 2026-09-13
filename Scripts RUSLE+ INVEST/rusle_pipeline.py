"""
RUSLE Soil Erosion Model Pipeline
=================================
A = R x K x LS x C x P

Run inside the 'rusle' conda environment:
    conda env create -f environment.yml
    conda activate rusle
    python rusle_pipeline.py --config config.yaml

Expected inputs (see config.yaml for paths):
  - DEM (GeoTIFF)                          -> used for LS-factor
  - Rainfall (CSV of monthly mm, OR a pre-computed R-factor value)
  - Sand / Clay / SOC rasters (GeoTIFF)    -> used for K-factor
  - Classified LULC raster (GeoTIFF)       -> used for C-factor (with lookup table)
  - AOI boundary (Shapefile/GeoJSON)       -> optional, used to mask outputs
  - P-factor raster (optional)             -> defaults to 1.0 everywhere

Output:
  - r_factor.tif, k_factor.tif, ls_factor.tif, c_factor.tif, p_factor.tif
  - soil_loss_A.tif   (final annual soil loss, t/ha/yr)
  - summary.txt       (stats + methodology notes)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask as rio_mask
from scipy import ndimage

try:
    import richdem as rd
except ImportError:
    rd = None

try:
    import geopandas as gpd
except ImportError:
    gpd = None

try:
    import yaml
except ImportError:
    yaml = None


NODATA_SENTINEL = 255  # OpenLandMap 8-bit fill value


def fill_nodata_nearest(arr):
    """
    Fill NaN gaps in a 2D array using nearest-valid-pixel interpolation.
    Used for the OpenLandMap SOC (organic carbon) band, which commonly has
    large real data gaps (e.g. ~55% NoData over young/dynamic char land).
    """
    if not np.any(np.isnan(arr)):
        return arr
    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        return arr  # nothing to fill from
    idx = ndimage.distance_transform_edt(~valid_mask, return_distances=False, return_indices=True)
    filled = arr[tuple(idx)]
    return filled


# ----------------------------------------------------------------------
# Utility: read / write / align rasters
# ----------------------------------------------------------------------

def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        profile = src.profile
        nodata = src.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, profile


def write_raster(path, arr, profile, dtype="float32", nodata=np.nan):
    out_profile = profile.copy()
    out_profile.update(dtype=dtype, count=1, nodata=nodata)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(arr.astype(dtype), 1)
    print(f"  wrote {path}")


def resample_to_reference(src_path, ref_profile):
    """Reproject/resample any raster onto the reference grid (usually the DEM grid)."""
    with rasterio.open(src_path) as src:
        dst_arr = np.full((ref_profile["height"], ref_profile["width"]), np.nan, dtype="float64")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=Resampling.bilinear,
        )
    return dst_arr


def clip_to_aoi(path, aoi_path):
    if gpd is None:
        raise RuntimeError("geopandas not installed; cannot clip to AOI")
    aoi = gpd.read_file(aoi_path)
    with rasterio.open(path) as src:
        aoi = aoi.to_crs(src.crs)
        out_arr, out_transform = rio_mask(src, aoi.geometry, crop=True, nodata=np.nan)
        out_profile = src.profile.copy()
        out_profile.update(height=out_arr.shape[1], width=out_arr.shape[2], transform=out_transform)
    return out_arr[0], out_profile


# ----------------------------------------------------------------------
# LS-factor  (Moore & Burch 1986 / Desmet & Govers 1996)
# ----------------------------------------------------------------------

def compute_ls_factor(dem_path, cell_size=30.0, slope_length_cap_m=300.0):
    if rd is None:
        raise RuntimeError("richdem not installed. `pip install richdem` inside the conda env.")

    print("[LS] loading DEM ...")
    with rasterio.open(dem_path) as src:
        dem_arr = src.read(1).astype("float64")
        dem_nodata = src.nodata
    if dem_nodata is not None:
        dem_arr = np.where(dem_arr == dem_nodata, np.nan, dem_arr)
    dem = rd.rdarray(dem_arr, no_data=np.nan)
    # richdem defaults to a 1x1 unit geotransform if none is set, which makes
    # its internal slope calculation treat each pixel as 1m apart instead of
    # the real ~30m spacing — massively inflating slope (and therefore LS).
    # NOTE: in-place operations like FillDepressions can strip this custom
    # attribute off the array, so we re-set it before every richdem call
    # that depends on real-world distances.
    geotransform = (0.0, cell_size, 0.0, 0.0, 0.0, -cell_size)
    dem.geotransform = geotransform

    print("[LS] filling depressions ...")
    rd.FillDepressions(dem, epsilon=True, in_place=True)
    dem.geotransform = geotransform  # re-apply: FillDepressions may reset it

    print("[LS] computing D8 flow accumulation ...")
    flow_acc = rd.FlowAccumulation(dem, method="D8")

    print("[LS] computing slope (degrees) ...")
    dem.geotransform = geotransform  # re-apply again, just before slope calc
    slope_deg = rd.TerrainAttribute(dem, attrib="slope_degrees")

    flow_acc = np.array(flow_acc, dtype="float64")
    slope_deg = np.array(slope_deg, dtype="float64")
    slope_rad = np.radians(slope_deg)
    print(f"[LS] slope stats (deg): min={np.nanmin(slope_deg):.3f} "
          f"max={np.nanmax(slope_deg):.3f} mean={np.nanmean(slope_deg):.3f}")

    # cap effective flow length (standard RUSLE practice on flat terrain)
    max_cells = slope_length_cap_m / cell_size
    flow_acc_capped = np.minimum(flow_acc, max_cells)

    ls = ((flow_acc_capped * cell_size) / 22.13) ** 0.4 * (np.sin(slope_rad) / 0.0896) ** 1.3
    ls = np.nan_to_num(ls, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[LS] LS-factor stats: min={np.nanmin(ls):.3f} max={np.nanmax(ls):.3f} mean={np.nanmean(ls):.3f}")

    with rasterio.open(dem_path) as src:
        profile = src.profile
    return ls, profile


# ----------------------------------------------------------------------
# R-factor  (Modified Fournier Index -> Arnoldus 1980 / Renard & Freimund 1994)
# ----------------------------------------------------------------------

def compute_r_factor_from_monthly_csv(csv_path):
    """csv columns: month, rainfall_mm  (12 rows, long-term monthly average)"""
    import csv as csv_module

    monthly = []
    with open(csv_path) as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            monthly.append(float(row["rainfall_mm"]))
    monthly = np.array(monthly)
    if len(monthly) != 12:
        print(f"  warning: expected 12 monthly values, got {len(monthly)}")

    P = monthly.sum()
    MFI = np.sum(monthly ** 2) / P
    R = 0.0483 * MFI ** 1.61
    print(f"[R] annual total = {P:.1f} mm, MFI = {MFI:.2f}, R = {R:.2f} MJ.mm/(ha.h.yr)")
    return R


def compute_r_factor_uniform(value, ref_profile):
    """Broadcast a single zonal R value across the whole grid."""
    arr = np.full((ref_profile["height"], ref_profile["width"]), value, dtype="float64")
    return arr


# ----------------------------------------------------------------------
# K-factor  (Williams 1995 EPIC formula)
# ----------------------------------------------------------------------

def compute_k_factor(sand_path, clay_path, soc_path, ref_profile):
    print("[K] loading & resampling sand/clay/SOC ...")

    def _load_and_mask(path):
        with rasterio.open(path) as src:
            arr = src.read(1).astype("float64")
        arr[arr == NODATA_SENTINEL] = np.nan
        return arr

    def _resample(arr, path, ref_profile):
        with rasterio.open(path) as src:
            src_transform, src_crs = src.transform, src.crs
        dst = np.full((ref_profile["height"], ref_profile["width"]), np.nan, dtype="float64")
        reproject(
            source=arr, destination=dst,
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"],
            resampling=Resampling.bilinear,
        )
        return dst

    sand_native = _load_and_mask(sand_path)
    clay_native = _load_and_mask(clay_path)
    soc_native = fill_nodata_nearest(_load_and_mask(soc_path))  # SOC often has real data gaps

    sand = _resample(sand_native, sand_path, ref_profile)
    clay = _resample(clay_native, clay_path, ref_profile)
    soc_raw = _resample(soc_native, soc_path, ref_profile)

    sand, clay, soc_raw = (fill_nodata_nearest(a) for a in (sand, clay, soc_raw))

    silt = 100.0 - sand - clay
    C = soc_raw / 5.0  # OpenLandMap SOC scale factor -> g/kg -> % organic carbon approx

    return _k_from_texture(sand, silt, clay, C)


def compute_k_factor_from_stack(soil_stack_path, ref_profile,
                                 band_order=("sand", "silt", "clay", "soc")):
    """
    Reads a single multi-band GeoTIFF (as produced by extract_soil_data.py,
    band order: sand, silt, clay, organic_carbon) and computes K directly.
    NoData masking and gap-filling happen at native resolution, BEFORE
    resampling, so the 255 sentinel never gets blended into real values
    by bilinear interpolation.
    """
    print(f"[K] loading soil property stack: {soil_stack_path} (bands={band_order})")
    with rasterio.open(soil_stack_path) as src:
        bands = {}
        for i, name in enumerate(band_order, start=1):
            arr = src.read(i).astype("float64")
            arr[arr == NODATA_SENTINEL] = np.nan
            bands[name] = arr
        src_profile = src.profile

    # Sand/Silt/Clay export bug: masked/invalid pixels came through as
    # literal 0 instead of the 255 sentinel. A real soil pixel can't have
    # sand=silt=clay=0 simultaneously (they should sum to ~100%), so treat
    # that triple-zero combination as NoData too.
    triple_zero = (bands["sand"] == 0) & (bands["silt"] == 0) & (bands["clay"] == 0)
    n_triple = np.sum(triple_zero)
    if n_triple > 0:
        print(f"[K] found {n_triple} sand=silt=clay=0 pixels (export bug, treating as NoData) ...")
        for name in ("sand", "silt", "clay"):
            bands[name][triple_zero] = np.nan

    # SOC commonly has large real data gaps (e.g. over young/dynamic char
    # land) — fill via nearest-valid-neighbor at native resolution first
    for name in ("sand", "silt", "clay", "soc"):
        n_gaps = np.sum(np.isnan(bands[name]))
        if n_gaps > 0:
            print(f"[K] filling {n_gaps} {name} NoData pixels (nearest-neighbor, native res) ...")
            bands[name] = fill_nodata_nearest(bands[name])

    def _resample_band(arr):
        # resample this band onto the reference grid using an in-memory reproject
        dst = np.full((ref_profile["height"], ref_profile["width"]), np.nan, dtype="float64")
        reproject(
            source=arr,
            destination=dst,
            src_transform=src_profile["transform"],
            src_crs=src_profile["crs"],
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=Resampling.bilinear,
        )
        return dst

    sand = _resample_band(bands["sand"])
    silt = _resample_band(bands["silt"])
    clay = _resample_band(bands["clay"])
    soc_raw = _resample_band(bands["soc"])

    # any remaining post-resample edge NaNs (e.g. outside the source extent)
    # are filled with nearest-neighbor too, so the output covers the full ref grid
    sand, silt, clay, soc_raw = (fill_nodata_nearest(a) for a in (sand, silt, clay, soc_raw))

    C = soc_raw / 5.0  # OpenLandMap SOC scale factor -> raw/5 = g/kg

    return _k_from_texture(sand, silt, clay, C)


def _k_from_texture(sand, silt, clay, C):
    """Williams (1995) EPIC K-factor formula, shared by both K-factor code paths."""

    SN1 = 1.0 - sand / 100.0
    f_csand = 0.2 + 0.3 * np.exp(-0.256 * sand * (1.0 - silt / 100.0))
    f_clsi = (silt / (clay + silt)) ** 0.3
    f_orgc = 1.0 - (0.25 * C) / (C + np.exp(3.72 - 2.95 * C))
    f_hisand = 1.0 - (0.7 * SN1) / (SN1 + np.exp(-5.51 + 22.9 * SN1))

    k_us = f_csand * f_clsi * f_orgc * f_hisand
    k_si = k_us * 0.1317  # convert to t.ha.h / (ha.MJ.mm)

    valid = np.sum(~np.isnan(k_si))
    total = k_si.size
    print(f"[K] valid pixels: {valid}/{total} (rest were NoData / outside AOI)")
    return k_si


# ----------------------------------------------------------------------
# C-factor  (classified LULC raster + lookup table)
# ----------------------------------------------------------------------

DEFAULT_C_LOOKUP = {
    0: 0.0,    # waterbody
    1: 0.6,    # bareland
    2: 0.05,   # builtup / settlement
    3: 0.02,   # vegetation
}


def compute_c_factor(lulc_path, ref_profile, lookup=None):
    """
    lulc_path may be a single-band classified raster, or a multi-band file
    (e.g. clip2015/2020/2025.tif) where band 1 holds the class codes.
    Only band 1 is read/resampled here — verified against a GRASS r.report
    ground-truth area check (see project notes), so no extra masking band
    is needed.
    """
    lookup = lookup or DEFAULT_C_LOOKUP
    print(f"[C] using lookup table: {lookup}")
    lulc = resample_to_reference(lulc_path, ref_profile)
    lulc_rounded = np.round(lulc)

    c = np.full(lulc.shape, np.nan, dtype="float64")
    for cls, cval in lookup.items():
        c[lulc_rounded == cls] = cval

    unmatched = np.sum(np.isnan(c) & ~np.isnan(lulc))
    if unmatched > 0:
        print(f"  warning: {unmatched} pixels had a class code not in the lookup table")
    return c


# ----------------------------------------------------------------------
# P-factor
# ----------------------------------------------------------------------

def compute_p_factor(ref_profile, p_path=None):
    if p_path is None:
        print("[P] no P-factor raster given -> assuming P = 1.0 everywhere")
        return np.ones((ref_profile["height"], ref_profile["width"]), dtype="float64")
    return resample_to_reference(p_path, ref_profile)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def load_config(path):
    if yaml is None:
        raise RuntimeError("pyyaml not installed. `pip install pyyaml`")
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="RUSLE soil erosion pipeline")
    parser.add_argument("--config", required=True, help="path to config.yaml")
    parser.add_argument("--outdir", default="rusle_outputs", help="output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. LS-factor: use a pre-computed raster if given, else derive from DEM
    print("\n=== LS-factor ===")
    if cfg.get("ls_factor_path"):
        print(f"[LS] using pre-computed raster: {cfg['ls_factor_path']}")
        ls, ref_profile = read_raster(cfg["ls_factor_path"])
    else:
        ls, ref_profile = compute_ls_factor(
            cfg["dem_path"],
            cell_size=cfg.get("dem_cell_size_m", 30.0),
            slope_length_cap_m=cfg.get("slope_length_cap_m", 300.0),
        )
    write_raster(outdir / "ls_factor.tif", ls, ref_profile)

    # 2. R-factor: use a pre-computed raster if given, else derive from rainfall
    print("\n=== R-factor ===")
    if cfg.get("r_factor_path"):
        print(f"[R] using pre-computed raster: {cfg['r_factor_path']}")
        r, r_profile = read_raster(cfg["r_factor_path"])
        r = resample_to_reference(cfg["r_factor_path"], ref_profile)
        r_value = float(np.nanmean(r))
    elif cfg.get("r_factor_value") is not None:
        r_value = cfg["r_factor_value"]
        print(f"[R] using pre-computed R-factor value = {r_value}")
        r = compute_r_factor_uniform(r_value, ref_profile)
    else:
        r_value = compute_r_factor_from_monthly_csv(cfg["rainfall_monthly_csv"])
        r = compute_r_factor_uniform(r_value, ref_profile)
    write_raster(outdir / "r_factor.tif", r, ref_profile)

    # 3. K-factor: use the 4-band soil stack (sand, silt, clay, soc) if given,
    #    else fall back to three separate single-band rasters
    print("\n=== K-factor ===")
    if cfg.get("soil_properties_path"):
        k = compute_k_factor_from_stack(
            cfg["soil_properties_path"], ref_profile,
            band_order=tuple(cfg.get("soil_band_order", ["sand", "silt", "clay", "soc"])),
        )
    else:
        k = compute_k_factor(cfg["sand_path"], cfg["clay_path"], cfg["soc_path"], ref_profile)
    write_raster(outdir / "k_factor.tif", k, ref_profile)

    # 4-8. C-factor + final A, once per LULC year
    # cfg["lulc_path"] can be either:
    #   - a single path (one C-factor / one A map), or
    #   - a dict of {year_label: path} for multiple years (2015/2020/2025 etc.)
    lulc_cfg = cfg["lulc_path"]
    if isinstance(lulc_cfg, dict):
        lulc_years = lulc_cfg
    else:
        lulc_years = {"": lulc_cfg}

    lookup = cfg.get("c_lookup", None)
    p = compute_p_factor(ref_profile, cfg.get("p_path"))
    write_raster(outdir / "p_factor.tif", p, ref_profile)

    all_summaries = []
    for year, lulc_path in lulc_years.items():
        suffix = f"_{year}" if year else ""
        print(f"\n=== C-factor {year or ''} ===".rstrip())
        c = compute_c_factor(lulc_path, ref_profile, lookup=lookup)
        write_raster(outdir / f"c_factor{suffix}.tif", c, ref_profile)

        print(f"\n=== Final soil loss A{('  ('+str(year)+')') if year else ''} = R x K x LS x C x P ===")
        A = r * k * ls * c * p
        write_raster(outdir / f"soil_loss_A{suffix}.tif", A, ref_profile)

        if cfg.get("aoi_path"):
            print(f"\n=== Clipping {year or 'output'} to AOI ===")
            for name in [f"c_factor{suffix}", f"soil_loss_A{suffix}"]:
                clipped, clipped_profile = clip_to_aoi(outdir / f"{name}.tif", cfg["aoi_path"])
                write_raster(outdir / f"{name}_clipped.tif", clipped, clipped_profile)

        valid_A = A[~np.isnan(A)]
        summary = (
            f"--- {year or 'RUSLE run'} ---\n"
            f"Pixels valid: {valid_A.size} / {A.size}\n"
            f"Min A (t/ha/yr):  {np.nanmin(A):.3f}\n"
            f"Max A (t/ha/yr):  {np.nanmax(A):.3f}\n"
            f"Mean A (t/ha/yr): {np.nanmean(A):.3f}\n"
        )
        print("\n" + summary)
        all_summaries.append(summary)

    # shared, non-yearly layers (LS, R, K only depend on static DEM/soil/rainfall)
    if cfg.get("aoi_path"):
        for name in ["r_factor", "k_factor", "ls_factor", "p_factor"]:
            clipped, clipped_profile = clip_to_aoi(outdir / f"{name}.tif", cfg["aoi_path"])
            write_raster(outdir / f"{name}_clipped.tif", clipped, clipped_profile)

    summary_text = f"RUSLE soil loss summary\nR-factor used: {r_value}\n\n" + "\n".join(all_summaries)
    print("\n" + summary_text)
    (outdir / "summary.txt").write_text(summary_text)


if __name__ == "__main__":
    sys.exit(main())
