"""
sensitivity_analysis.py
========================
Tests how sensitive the RUSLE soil-loss result is to +/-10% uncertainty in
the R-factor (rainfall erosivity) and K-factor (soil erodibility) inputs.

Reuses the LS-factor, C-factor, and P-factor already computed by
rusle_pipeline.py (in rusle_outputs/) — it does NOT re-run richdem or
re-derive K from the soil stack, so this runs in seconds, not minutes.

Run inside the 'rusle' conda environment, after rusle_pipeline.py has
already been run at least once (so rusle_outputs/ exists):

    python sensitivity_analysis.py --outdir rusle_outputs --years 2015 2020 2025

Output:
    sensitivity_outputs/sensitivity_summary.csv   - full results table
    sensitivity_outputs/sensitivity_summary.png    - bar chart
    Also prints a formatted summary table to the terminal.
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio


def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        nodata = src.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    return arr


def pixel_area_ha(path):
    """
    Returns the pixel area in hectares. Handles both:
    - Projected/metric CRS rasters (pixel size already in meters), and
    - Geographic CRS rasters (EPSG:4326, pixel size in degrees) — this is
      the case for rusle_outputs/*.tif, which inherit the DEM's original
      geographic transform. Degree pixel sizes are converted to meters
      using the raster's center latitude (longitude degrees shrink by
      cos(latitude); latitude degrees are ~constant, ~111.32 km/degree).
    """
    with rasterio.open(path) as src:
        transform = src.transform
        crs = src.crs
        bounds = src.bounds

    px_w = abs(transform.a)
    px_h = abs(transform.e)

    if crs is not None and crs.is_geographic:
        center_lat = (bounds.top + bounds.bottom) / 2.0
        m_per_deg_lat = 110574.0
        m_per_deg_lon = 111320.0 * np.cos(np.radians(center_lat))
        px_w_m = px_w * m_per_deg_lon
        px_h_m = px_h * m_per_deg_lat
        print(f"  (raster CRS is geographic; converted {px_w:.6f}x{px_h:.6f} deg "
              f"-> {px_w_m:.2f}x{px_h_m:.2f} m at latitude {center_lat:.3f})")
        return (px_w_m * px_h_m) / 10000.0

    return (px_w * px_h) / 10000.0


def compute_total_and_mean(A, area_ha):
    valid = A[~np.isnan(A)]
    total_tons = np.nansum(A) * area_ha
    mean_a = np.nanmean(A)
    max_a = np.nanmax(A) if valid.size else np.nan
    return total_tons, mean_a, max_a


def main():
    parser = argparse.ArgumentParser(description="RUSLE sensitivity analysis (+/-10% R and K)")
    parser.add_argument("--outdir", default="rusle_outputs", help="RUSLE pipeline output folder (input here)")
    parser.add_argument("--years", nargs="+", default=["2015", "2020", "2025"])
    parser.add_argument("--variation", type=float, default=0.10, help="fractional variation, default 0.10 = +/-10%%")
    parser.add_argument("--savedir", default="sensitivity_outputs")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    savedir = Path(args.savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    ls_path = outdir / "ls_factor.tif"
    r_path = outdir / "r_factor.tif"
    k_path = outdir / "k_factor.tif"
    p_path = outdir / "p_factor.tif"

    print("Loading static factors (LS, R, K, P) ...")
    ls = read_raster(ls_path)
    r_base = read_raster(r_path)
    k_base = read_raster(k_path)
    p = read_raster(p_path)
    area_ha = pixel_area_ha(ls_path)
    print(f"  pixel area = {area_ha:.5f} ha")

    v = args.variation
    scenarios = {
        "Baseline": (1.0, 1.0),
        f"R +{int(v*100)}%": (1 + v, 1.0),
        f"R -{int(v*100)}%": (1 - v, 1.0),
        f"K +{int(v*100)}%": (1.0, 1 + v),
        f"K -{int(v*100)}%": (1.0, 1 - v),
    }

    rows = []
    for year in args.years:
        c_path = outdir / f"c_factor_{year}.tif"
        if not c_path.exists():
            c_path = outdir / "c_factor.tif"
        c = read_raster(c_path)

        print(f"\n=== {year} ===")
        baseline_total = None
        for label, (r_mult, k_mult) in scenarios.items():
            r = r_base * r_mult
            k = k_base * k_mult
            A = r * k * ls * c * p
            total_tons, mean_a, max_a = compute_total_and_mean(A, area_ha)
            if label == "Baseline":
                baseline_total = total_tons
            pct_change = 0.0 if baseline_total is None else 100 * (total_tons - baseline_total) / baseline_total
            rows.append({
                "year": year, "scenario": label,
                "total_tons_per_yr": total_tons, "mean_A_t_ha_yr": mean_a,
                "max_A_t_ha_yr": max_a, "pct_change_vs_baseline": pct_change,
            })
            print(f"  {label:12s}  total={total_tons:9.1f} t/yr   mean={mean_a:.4f} t/ha/yr   "
                  f"change={pct_change:+.1f}%")

    # ---- write CSV ----
    import csv
    csv_path = savedir / "sensitivity_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {csv_path}")

    # ---- chart ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        years = args.years
        scenario_labels = list(scenarios.keys())
        fig, ax = plt.subplots(figsize=(9, 5.5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        width = 0.15
        x = np.arange(len(years))
        colors = ["#2c3e50", "#27ae60", "#16a085", "#2980b9", "#8e44ad"]
        for i, label in enumerate(scenario_labels):
            vals = [r["total_tons_per_yr"] for r in rows if r["scenario"] == label]
            ax.bar(x + (i - 2) * width, vals, width=width, label=label,
                   color=colors[i % len(colors)], edgecolor="black", linewidth=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels(years, fontsize=11, fontweight="bold")
        ax.set_ylabel("Total soil loss (t/yr)")
        ax.set_title(f"RUSLE Sensitivity Analysis — +/-{int(v*100)}% R-factor and K-factor", fontsize=13, fontweight="bold")
        ax.legend(fontsize=8.5, frameon=True, edgecolor="black")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color("black"); spine.set_linewidth(1.0)
        plt.tight_layout()
        chart_path = savedir / "sensitivity_summary.png"
        plt.savefig(chart_path, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {chart_path}")
    except ImportError:
        print("matplotlib not available; skipped chart generation")

    print("\nDone. Key takeaway to check: does the YEAR-TO-YEAR ranking "
          "(e.g. 2020 lowest) stay the same across every scenario row above? "
          "If yes, the LULC-driven trend is robust to R/K uncertainty.")


if __name__ == "__main__":
    main()
