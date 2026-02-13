# FDL-X 2024 -- Radiation Team -- ML Experiments

LSTM-based models for predicting space radiation during Solar Proton Events (SPEs), developed as part of FDL-X 2024 (Frontier Development Lab).

## Overview

This project trains recurrent neural networks to forecast radiation levels in deep space using solar observation data. Two model architectures are available:

- **RadRecurrent** -- LSTM model using time-series inputs only (GOES X-ray, GOES proton flux, BioSentinel dose rate)
- **RadRecurrentWithSDO** -- Extends RadRecurrent with SDO/AIA solar imagery as additional context via a CNN embedding

Uncertainty is estimated via Monte Carlo dropout at inference time.

## External Data Sources

| Source | Description | Script |
|--------|-------------|--------|
| GOES XRS | X-ray flux (GOES-16, 1-min avg) | `data_scripts/get_goes_xrs.py` |
| GOES SGPS | Solar & galactic proton flux >10 and >100 MeV (GOES-16, 1-min avg) | `data_scripts/get_goes_sgps.py` |
| RSTN Radio | Solar radio burst data at 8 frequencies (Sagamore Hill, 1-sec resampled to 1-min) | `data_scripts/get_rstn_radio.py` |
| SDOML-lite | SDO/AIA and HMI imagery (6 channels, 512x512, 15-min cadence) | Provided externally |
| RadLab | BioSentinel BPD and CRaTER-D1D2 absorbed dose rates | Provided externally (DuckDB) |

Data download scripts support parallel workers and multi-node distribution. Processing scripts (`data_scripts/process_*.py`) convert raw NetCDF/gzip files to CSV.

## Project Structure

```
data_scripts/          # Download and process raw data
  get_goes_sgps.py     # Download GOES SGPS NetCDF files from NOAA
  get_goes_xrs.py      # Download GOES XRS NetCDF files from NOAA
  get_rstn_radio.py    # Download RSTN radio data from NOAA
  process_goes_sgps.py # Process SGPS NetCDF to CSV (>10 MeV, >100 MeV)
  process_goes_xrs.py  # Process XRS NetCDF to CSV
  process_rstn_radio.py# Process RSTN gzip to CSV (8 freq, 1-min resample)
scripts/
  models.py            # Model definitions (SDOEmbedding, RadRecurrent, RadRecurrentWithSDO)
  datasets.py          # PyTorch datasets (GOESXRS, GOESSGPS, RSTNRadio, RadLab, SDOMLlite, Sequences)
  events.py            # Solar Proton Event catalog (BioSentinel & CRaTER epochs)
  run.py               # Training and testing entry point
  event_plot.py        # Animated event visualization (SDO + time series)
  data_stats.py        # Dataset statistics and histograms
Dockerfile             # PyTorch 2.4.0 / CUDA 11.8 runtime
```

## Usage

### Download and process data

```bash
python data_scripts/get_goes_xrs.py --target_dir ./data/goes-xrs
python data_scripts/process_goes_xrs.py --source_dir ./data/goes-xrs --target_file ./data/goes/goes-xrs.csv
```

### Train

```bash
python scripts/run.py \
  --mode train \
  --data_dir ./data \
  --target_dir ./results \
  --model_type RadRecurrentWithSDO \
  --device cuda \
  --epochs 100
```

### Test

```bash
python scripts/run.py \
  --mode test \
  --data_dir ./data \
  --target_dir ./results \
  --model_file ./results/epoch-100-model.pth \
  --device cuda
```

## Dependencies

- PyTorch >= 2.4
- pandas, numpy, matplotlib
- netCDF4, xarray (data processing)
- duckdb (RadLab database)
- sunpy (SDO colormaps)
- tqdm
