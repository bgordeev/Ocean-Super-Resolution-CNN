# Ocean SST Super-Resolution with CNNs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Convolutional Neural Network (CNN) approach for enhancing the spatial resolution of ocean Sea Surface Temperature (SST) satellite data. This project implements a super-resolution model that upscales low-resolution (0.25°) SST measurements to high-resolution (0.05°) predictions.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

Ocean surface temperature prediction is crucial for understanding climate dynamics, marine ecosystems, and weather patterns. Traditional satellite measurements often have limited spatial resolution, which can miss important fine-scale temperature variations caused by eddies, currents, and coastal processes.

This project leverages Convolutional Neural Networks to perform **5× super-resolution** on SST data, transforming 40×40 pixel low-resolution patches into detailed 200×200 pixel high-resolution predictions.

### Research Question

> *To what extent can Convolutional Neural Networks (CNNs) improve the accuracy and reliability of ocean surface temperature prediction models?*

## Features

- **Data Pipeline**: Automated downloading and preprocessing of MUR SST satellite data
- **Patch Extraction**: Intelligent extraction of training patches with land masking
- **CNN Model**: Custom super-resolution architecture with skip connections
- **Training Framework**: Complete training loop with checkpointing and visualization
- **Evaluation Tools**: Visual comparison and quality assessment utilities

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- NASA Earthdata account (for data download)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ocean-sst-superres.git
   cd ocean-sst-superres
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Earthdata credentials** (for data download)
   ```bash
   # Create a .env file or set environment variables
   export EARTHDATA_USERNAME="your_username"
   export EARTHDATA_PASSWORD="your_password"
   ```

## Data

This project uses the **Multi-scale Ultra-high Resolution (MUR) SST** dataset from NASA's Physical Oceanography Distributed Active Archive Center (PO.DAAC).

### Dataset Specifications

| Dataset | Resolution | Grid Size | Source |
|---------|------------|-----------|--------|
| MUR v4.1 (High-res) | 0.01° | ~0.01° globally | Coarsened to 0.05° |
| MUR25 v4.2 (Low-res) | 0.25° | ~25 km globally | Used as input |

### Downloading Data

```bash
# Download data for a specific date range
python src/data/download.py --start-date 2021-01-01 --end-date 2021-01-31
```

### Data Preprocessing

The preprocessing pipeline consists of two stages:

1. **Coarsening**: Reduce 0.01° data to 0.05° for manageable high-res targets
   ```bash
   python src/data/coarsen.py --input-dir data/raw --output-dir data/coarsened
   ```

2. **Patch Extraction**: Create aligned patch pairs for training
   ```bash
   python src/data/patches.py --high-res-dir data/coarsened --low-res-dir data/raw --output-dir data/patches
   ```

## Usage

### Training

```bash
# Train the model with default parameters
python src/train.py

# Train with custom parameters
python src/train.py \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --data-dir data/patches
```

### Inference

```bash
# Run inference on a single image
python src/inference.py --input path/to/low_res_image.png --output path/to/output.png

# Run inference on a directory
python src/inference.py --input-dir data/test/low_res --output-dir results/
```

### Jupyter Notebook

For interactive exploration, use the provided notebook:

```bash
jupyter notebook notebooks/ocean_sst_superres.ipynb
```

## Model Architecture

The super-resolution CNN uses the following architecture:

```
Input (3×40×40) 
    │
    ▼
Bilinear Upsample (5×) → (3×200×200)
    │
    ▼
Conv2D (3 to 64, 3×3) + ReLU
    │
    ▼
Conv2D (64 to 128, 3×3) + ReLU
    │
    ├─────────────────┐
    ▼                 │ Skip Connection
Conv2D (128 to 64, 3×3) + ReLU ←──┘
    │
    ├───────────────────┐
    ▼                   │ Skip Connection
Conv2D (64to 3, 3×3) ←──┘ (from upsampled input)
    │
    ▼
Output (3×200×200)
```

### Key Features

- **Initial Upsampling**: Bilinear interpolation provides a strong baseline
- **Skip Connections**: Preserve low-level features and improve gradient flow
- **Residual Learning**: Model learns to add fine details to the upsampled image

## Results

The model demonstrates the ability to enhance SST imagery, though some limitations remain:

### Visual Comparison

| Low Resolution (40×40) | Ground Truth (200×200) | Model Output (200×200) |
|:----------------------:|:----------------------:|:----------------------:|
| ![Low Res](docs/images/low_res_example.png) | ![Ground Truth](docs/images/high_res_example.png) | ![Output](docs/images/output_example.png) |

### Metrics

| Metric | Value |
|--------|-------|
| MSE Loss | ~0.005 |
| Training Time | ~10 epochs |

### Known Limitations

- **Blurriness**: Some loss of fine detail in upscaled images
- **Edge Artifacts**: Minor distortions at image boundaries
- **Data Dependency**: Performance varies with ocean region and conditions

## Project Structure

```
ocean-sst-superres/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation
├── .gitignore               # Git ignore rules
├── configs/
│   └── default.yaml         # Default configuration
├── data/                    # Data directory (not tracked)
│   ├── raw/                 # Raw downloaded data
│   ├── coarsened/           # Coarsened high-res data
│   └── patches/             # Extracted patches
├── docs/
│   ├── DATA.md              # Data documentation
│   ├── MODEL.md             # Model documentation
│   └── images/              # Documentation images
├── notebooks/
│   └── ocean_sst_superres.ipynb  # Interactive notebook
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py      # Data download utilities
│   │   ├── coarsen.py       # Coarsening functions
│   │   └── patches.py       # Patch extraction
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn.py           # CNN model definition
│   ├── utils/
│   │   ├── __init__.py
│   │   └── visualization.py # Plotting utilities
│   ├── train.py             # Training script
│   └── inference.py         # Inference script
└── tests/
    ├── __init__.py
    └── test_model.py        # Model tests
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **NASA PO.DAAC** for providing the MUR SST dataset
- **PyTorch** team for the deep learning framework
- **xarray** developers for the netCDF handling library

## References

1. Chin, T. M., Vazquez-Cuervo, J., & Armstrong, E. M. (2017). A multi-scale high-resolution analysis of global sea surface temperature. *Remote Sensing of Environment*, 200, 154-169.

2. Ducournau, A., & Fablet, R. (2017). Deep learning for ocean remote sensing: An application of convolutional neural networks for super-resolution on satellite-derived SST data. *IAPR Workshop on Pattern Recognition in Remote Sensing*.

3. Bolton, T., & Zanna, L. (2019). Applications of Deep Learning to Ocean Data Inference and Subgrid Parameterization. *Journal of Advances in Modeling Earth Systems*, 11(1), 376-399.

---

<p align="center">
  <i>Built with love for oceanographic research</i>
</p>
