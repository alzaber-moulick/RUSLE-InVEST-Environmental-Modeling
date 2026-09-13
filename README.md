# RUSLE-InVEST-Environmental-Modeling

Python workflows for soil erosion risk modeling on Bhasan Char, Bangladesh, using the **RUSLE** (Revised Universal Soil Loss Equation) and **InVEST SDR** (Sediment Delivery Ratio) frameworks, cross-validated against each other and against published literature.

## What This Project Does

Estimates annual soil loss (t/ha/yr) for 2015, 2020, and 2025 by combining:
- A Digital Elevation Model (Copernicus GLO-30) → slope & flow-length (LS-factor)
- Rainfall data (CHIRPS) → rainfall erosivity (R-factor)
- Soil texture/organic carbon (OpenLandMap) → soil erodibility (K-factor)
- Classified land-use/land-cover rasters (Sentinel-2, 2015/2020/2025) → cover-management (C-factor)

The same inputs are then run through **InVEST's Sediment Delivery Ratio (SDR)** model as an independent cross-check, distinguishing gross erosion from sediment actually reaching the watershed outlet.

## Repository Contents

| File | Purpose |
|---|---|
| `scripts/rusle_pipeline.py` | Core RUSLE calculation — computes R, K, LS, C, P factors and final soil loss per year |
| `scripts/make_maps.py` | Renders soil-loss GeoTIFF outputs as classified PNG maps |
| `scripts/sensitivity_analysis.py` | Tests how ±10% changes in R-factor/K-factor affect the final result |
| `scripts/invest_sdr.py` | Runs the InVEST SDR model, reusing RUSLE's DEM/R/K/LULC inputs |
| `scripts/biophysical_table.csv` | LULC class → C-factor/P-factor lookup table used by InVEST |
| `config.yaml` | RUSLE pipeline configuration (edit paths before running) |
| `environment.yml` | Conda environment for the RUSLE pipeline |
| `invest_environment.yml` | Conda environment for the InVEST SDR pipeline |

## Methodology Summary

**RUSLE**: A = R × K × LS × C × P, computed per-pixel. R, K, and LS are held constant across all three years; only the C-factor (from each year's classified LULC raster) varies, isolating the effect of land-cover change on erosion risk.

**InVEST SDR**: Run on the same DEM, R-factor, K-factor, and LULC inputs (reprojected to a projected UTM CRS, since SDR requires metric pixel spacing) to independently verify both the erosion magnitude and spatial pattern, and to additionally estimate how much eroded sediment is retained on the island versus exported to the surrounding water.

## Key Finding

Both models independently identify **2020 as the lowest-erosion year**, coinciding with peak vegetation cover following island afforestation, and **2015 as the highest**, when bareland dominated the island's surface. This land-cover-driven trend is robust to ±10% uncertainty in the R-factor and K-factor inputs (see `sensitivity_analysis.py`).

## Setup

```bash
# RUSLE
conda env create -f environment.yml
conda activate rusle
python scripts/rusle_pipeline.py --config config.yaml --outdir rusle_outputs
python scripts/make_maps.py --outdir rusle_outputs --years 2015 2020 2025
python scripts/sensitivity_analysis.py --outdir rusle_outputs --years 2015 2020 2025

# InVEST SDR
conda env create -f invest_environment.yml
conda activate invest
python scripts/invest_sdr.py
```

Note: raw input rasters (DEM, rainfall, soil, LULC, AOI boundary) are not included in this repository. Place your own copies in a local `data/` folder before running either pipeline, matching the paths referenced in `config.yaml`.

## Status

This is an active research project. A manuscript based on this work is in preparation; full result outputs and the detailed methodology report will be shared publicly following publication.

## License

MIT — see `LICENSE`.
