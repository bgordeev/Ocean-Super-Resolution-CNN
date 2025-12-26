# Data Documentation

This document describes the data used for training the SST super-resolution model.

## Dataset Overview

The model is trained on the **Multi-scale Ultra-high Resolution (MUR) Sea Surface Temperature** dataset from NASA's Physical Oceanography Distributed Active Archive Center (PO.DAAC).

### Data Sources

| Dataset | Version | Resolution | Coverage | Source |
|---------|---------|------------|----------|--------|
| MUR SST | v4.1 | 0.01° (~1 km) | Global | [PO.DAAC](https://podaac.jpl.nasa.gov/dataset/MUR-JPL-L4-GLOB-v4.1) |
| MUR25 SST | v4.2 | 0.25° (~25 km) | Global | [PO.DAAC](https://podaac.jpl.nasa.gov/dataset/MUR25-JPL-L4-GLOB-v04.2) |

### Data Format

The data is provided in NetCDF format with the following structure:

```
Dimensions:
  - time: 1 (daily)
  - lat: latitude points
  - lon: longitude points

Variables:
  - analysed_sst: Sea surface temperature (Kelvin)
  - analysis_error: Error estimate
  - mask: Land/ice/sea mask
  - sea_ice_fraction: Sea ice concentration
```

## Data Pipeline

### 1. Downloading Data

Data is downloaded from NASA's Earthdata archive. You need a free Earthdata account.

```bash
# Set credentials
export EARTHDATA_USERNAME="your_username"
export EARTHDATA_PASSWORD="your_password"

# Download data
python src/data/download.py --start-date 2021-01-01 --end-date 2021-01-31
```

### 2. Coarsening High-Resolution Data

The 0.01° data is coarsened to 0.05° to create manageable high-resolution targets:

```bash
python src/data/coarsen.py \
    --input-dir data/raw/high_res \
    --output-dir data/coarsened \
    --start-date 2021-01-01 \
    --end-date 2021-01-31
```

This reduces data size while maintaining enough detail for training.

### 3. Patch Extraction

Training patches are extracted from aligned high and low resolution images:

```bash
python src/data/patches.py \
    --high-res-dir data/coarsened \
    --low-res-dir data/raw/low_res \
    --output-dir data/patches \
    --start-date 2021-01-01 \
    --end-date 2021-01-31
```

**Patch specifications:**
- High-resolution: 200×200 pixels (0.05° per pixel)
- Low-resolution: 40×40 pixels (0.25° per pixel)
- Scale factor: 5×

**Land masking:**
Only patches containing no NaN values (ocean only) are kept.

## Data Structure

After preprocessing, the data directory should look like:

```
data/
├── raw/
│   ├── high_res/
│   │   └── YYYYMMDD_high_res.nc
│   └── low_res/
│       └── YYYYMMDD_low_res.nc
├── coarsened/
│   └── YYYYMMDD/
│       └── coarsened.nc
└── patches/
    ├── high_res/
    │   └── YYYYMMDD/
    │       └── patch_X_Y.png
    └── low_res/
        └── YYYYMMDD/
            └── patch_X_Y.png
```


