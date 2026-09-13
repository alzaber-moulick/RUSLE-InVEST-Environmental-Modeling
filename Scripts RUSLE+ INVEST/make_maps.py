"""
make_maps.py
============
Turns the RUSLE pipeline's soil_loss_A_*.tif outputs into colored,
legend-labeled PNG maps (erosion severity classes), similar in spirit
to a standard LULC map figure.

Run after rusle_pipeline.py, inside the same 'rusle' conda environment:

    python make_maps.py --outdir rusle_outputs --years 2015 2020 2025
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

# Erosion severity classes (t/ha/yr) — recalibrated for the corrected,
# physically-realistic RUSLE output on this near-flat island (typical range
# is now roughly 0-30 t/ha/yr, not the 0-800+ range from before the
# geotransform bugfix).
SEVERITY_BINS = [0, 1, 3, 8, np.inf]
SEVERITY_LABELS = ["Slight (0-1)", "Moderate (1-3)", "High (3-8)", "Severe (>8)"]
# Vivid, saturated colors that stand out clearly against a white background
SEVERITY_COLORS = ["#5d6d7e", "#27ae60", "#2980b9", "#e74c3c"]  # dark grey, green, blue, red


def classify(A):
    classed = np.digitize(A, SEVERITY_BINS[1:-1])  # 0..4
    classed = classed.astype(float)
    classed[np.isnan(A)] = np.nan
    return classed


def plot_one(path, title, out_png):
    with rasterio.open(path) as src:
        A = src.read(1).astype("float64")
        nodata = src.nodata
    if nodata is not None:
        A = np.where(A == nodata, np.nan, A)

    classed = classify(A)
    cmap = ListedColormap(SEVERITY_COLORS)
    cmap.set_bad(color="white")  # NaN / outside-AOI pixels render as clean white
    bounds = [i - 0.5 for i in range(len(SEVERITY_COLORS) + 1)]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.imshow(classed, cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    legend_handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.5, label=l)
                       for c, l in zip(SEVERITY_COLORS, SEVERITY_LABELS)]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=9,
              title="Soil loss (t/ha/yr)", frameon=True, facecolor="white", edgecolor="grey")

    valid = A[~np.isnan(A)]
    stats_txt = f"mean={np.nanmean(A):.1f}  max={np.nanmax(A):.1f} t/ha/yr" if valid.size else "no data"
    ax.text(0.99, 0.01, stats_txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="white",
            bbox=dict(facecolor="black", alpha=0.6, boxstyle="round"))

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_png}")


def plot_panel(paths_by_year, out_png):
    n = len(paths_by_year)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    cmap = ListedColormap(SEVERITY_COLORS)
    cmap.set_bad(color="white")
    bounds = [i - 0.5 for i in range(len(SEVERITY_COLORS) + 1)]
    norm = BoundaryNorm(bounds, cmap.N)

    for ax, (year, path) in zip(axes, paths_by_year.items()):
        with rasterio.open(path) as src:
            A = src.read(1).astype("float64")
            nodata = src.nodata
        if nodata is not None:
            A = np.where(A == nodata, np.nan, A)
        classed = classify(A)
        ax.set_facecolor("white")
        ax.imshow(classed, cmap=cmap, norm=norm)
        ax.set_title(f"Bhasan Char {year}", fontsize=13, fontweight="bold")
        ax.axis("off")

    legend_handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.5, label=l)
                       for c, l in zip(SEVERITY_COLORS, SEVERITY_LABELS)]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(SEVERITY_COLORS), fontsize=9,
               title="Soil loss (t/ha/yr)", bbox_to_anchor=(0.5, -0.05),
               frameon=True, facecolor="white", edgecolor="grey")
    fig.suptitle("RUSLE Soil Erosion Risk — Bhasan Char", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Render RUSLE soil-loss GeoTIFFs as colored maps")
    parser.add_argument("--outdir", default="rusle_outputs", help="pipeline output folder (input here)")
    parser.add_argument("--mapdir", default="rusle_maps", help="where to save PNG maps")
    parser.add_argument("--years", nargs="*", default=None,
                         help="year labels to render, e.g. 2015 2020 2025. "
                              "Omit if you only have soil_loss_A.tif (single run, no year suffix).")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    mapdir = Path(args.mapdir)
    mapdir.mkdir(parents=True, exist_ok=True)

    if args.years:
        paths_by_year = {}
        for year in args.years:
            # prefer the AOI-clipped raster so the map shows the true island
            # shape (white outside it) instead of the full rectangular grid
            tif = outdir / f"soil_loss_A_{year}_clipped.tif"
            if not tif.exists():
                tif = outdir / f"soil_loss_A_{year}.tif"
            if not tif.exists():
                print(f"  warning: no soil_loss_A file found for {year}, skipping")
                continue
            paths_by_year[year] = tif
            plot_one(tif, f"Bhasan Char Soil Erosion Risk — {year}",
                      mapdir / f"soil_loss_{year}.png")

        if len(paths_by_year) > 1:
            plot_panel(paths_by_year, mapdir / "soil_loss_all_years.png")
    else:
        tif = outdir / "soil_loss_A_clipped.tif"
        if not tif.exists():
            tif = outdir / "soil_loss_A.tif"
        plot_one(tif, "Bhasan Char Soil Erosion Risk", mapdir / "soil_loss.png")

    print(f"\nDone. Maps saved in: {mapdir}/")


if __name__ == "__main__":
    main()
