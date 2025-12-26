# Model Documentation

This document describes the neural network architectures used for SST super-resolution.

## Overview

The project implements three CNN architectures for 5× super-resolution:

1. **Basic CNN** - Simple architecture with skip connections
2. **Enhanced CNN** - Deeper network with residual blocks
3. **Sub-Pixel CNN** - Efficient architecture using pixel shuffle

## Basic Super-Resolution CNN

### Architecture

```
Input (3×40×40)
    │
    ▼
Bilinear Upsample (5×)
    │ → (3×200×200)
    ▼
Conv2D (3 to 64, 3×3) + ReLU
    │
    ├─────────────────┐
    ▼                 │
Conv2D (64 to 128, 3×3) + ReLU
    │                 │
    ▼                 │ Skip
Conv2D (128 to 64, 3×3) + ReLU ←─┘
    │
    ├─────────────────┐
    ▼                 │ Skip (from upsampled)
Conv2D (64 to 3, 3×3) ←──┘
    │
    ▼
Output (3×200×200)
```

### Key Features

- **Initial upsampling**: Uses bilinear interpolation to create a strong baseline
- **Skip connections**: Help preserve low-frequency information
- **Residual learning**: Network learns to add fine details to the upsampled image

### Usage

```python
from models.cnn import SuperResolutionCNN

model = SuperResolutionCNN(
    in_channels=3,
    scale_factor=5,
    base_filters=64
)
```

## Enhanced Super-Resolution CNN

### Architecture

Uses residual blocks for better feature extraction:

```
Input → Conv (9×9) → [Residual Block] × N → Conv → Upsample → Conv (9×9) → Output
                            ↓
                    Global Skip Connection
```

Each residual block:
```
Input → Conv → BN → ReLU → Conv → BN → + → ReLU → Output
   └───────────────────────────────────┘
```

### Key Features

- **Residual blocks**: Enable training of deeper networks
- **Batch normalization**: Stabilizes training
- **Global skip connection**: Preserves input information
- **Pixel shuffle upsampling**: More efficient than deconvolution

### Usage

```python
from models.cnn import EnhancedSuperResolutionCNN

model = EnhancedSuperResolutionCNN(
    in_channels=3,
    scale_factor=5,
    base_filters=64,
    num_residual_blocks=8
)
```

## Sub-Pixel CNN (ESPCN)

Based on the Efficient Sub-Pixel Convolutional Neural Network.

### Architecture

```
Input (3×40×40)
    │
    ▼
Conv2D (3 to 64, 5×5) + ReLU
    │
    ▼
Conv2D (64 to 64, 3×3) + ReLU
    │
    ▼
Conv2D (64 to 64, 3×3) + ReLU
    │
    ▼
Conv2D (64 to 3×25, 3×3)
    │
    ▼
PixelShuffle (5×)
    │
    ▼
Output (3×200×200)
```

### Key Features

- **No upsampling until the end**: All processing at low resolution
- **Pixel shuffle**: Rearranges channels into spatial dimensions
- **Efficient**: Much faster than interpolation-based methods

### Usage

```python
from models.cnn import SubPixelCNN

model = SubPixelCNN(
    in_channels=3,
    scale_factor=5,
    base_filters=64
)
```

## Model Selection

| Model | Parameters | Speed | Quality | Best For |
|-------|------------|-------|---------|----------|
| Basic | ~150K | Fast | Good | Quick experiments |
| Enhanced | ~1.5M | Medium | Better | Final training |
| Sub-Pixel | ~100K | Fastest | Good | Real-time inference |

## Training Details

### Loss Function

Mean Squared Error (MSE) is used as the primary loss:

```python
criterion = nn.MSELoss()
```

MSE directly optimizes for pixel-wise accuracy but may produce slightly blurry results.

### Optimizer

Adam optimizer with default parameters:

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### Hyperparameters

| Parameter | Default | Range |
|-----------|---------|-------|
| Learning rate | 0.001 | 0.0001-0.01 |
| Batch size | 32 | 8-64 |
| Epochs | 10 | 10-100 |
| Base filters | 64 | 32-128 |


