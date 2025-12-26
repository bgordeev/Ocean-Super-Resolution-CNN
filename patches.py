"""
Patch extraction utilities for SST super-resolution training.

This module provides functions to extract aligned patches from high-resolution
and low-resolution SST data for training the super-resolution CNN model.

Patches are extracted only from ocean regions (areas without NaN values from
land masking).
"""

import os
import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def extract_patches(
    high_res_data: np.ndarray,
    low_res_data: np.ndarray,
    high_res_patch_size: int = 200,
    low_res_patch_size: int = 40,
    stride: int = 200
) -> Tuple[list, list]:
    """
    Extract aligned patches from high and low resolution data.
    
    Patches are only extracted if they contain no NaN values (i.e., no land).
    The scale factor between high and low resolution is automatically computed.
    
    Args:
        high_res_data: High-resolution SST data array (H x W).
        low_res_data: Low-resolution SST data array.
        high_res_patch_size: Size of high-resolution patches.
        low_res_patch_size: Size of low-resolution patches.
        stride: Step size for patch extraction.
    
    Returns:
        Tuple of (high_res_patches, low_res_patches) lists.
    """
    scale_factor = high_res_patch_size // low_res_patch_size
    
    high_res_patches = []
    low_res_patches = []
    
    for i in range(0, high_res_data.shape[0] - high_res_patch_size, stride):
        for j in range(0, high_res_data.shape[1] - high_res_patch_size, stride):
            # Extract high-resolution patch
            hr_patch = high_res_data[i:i + high_res_patch_size, 
                                     j:j + high_res_patch_size]
            
            # Extract corresponding low-resolution patch
            lr_i = i // scale_factor
            lr_j = j // scale_factor
            lr_patch = low_res_data[lr_i:lr_i + low_res_patch_size,
                                    lr_j:lr_j + low_res_patch_size]
            
            # Only keep patches without NaN values (ocean only)
            if np.isnan(hr_patch).sum() == 0 and np.isnan(lr_patch).sum() == 0:
                high_res_patches.append(hr_patch)
                low_res_patches.append(lr_patch)
    
    return high_res_patches, low_res_patches


def save_patches(
    ds_high_res: xr.Dataset,
    ds_low_res: xr.Dataset,
    output_dir: str,
    date_str: str,
    high_res_patch_size: int = 200,
    low_res_patch_size: int = 40,
    stride: int = 200,
    colormap: str = 'viridis'
) -> Tuple[int, int]:
    """
    Extract and save patches as image files.
    
    Args:
        ds_high_res: High-resolution xarray Dataset.
        ds_low_res: Low-resolution xarray Dataset.
        output_dir: Base directory for output patches.
        date_str: Date string for directory naming.
        high_res_patch_size: Size of high-resolution patches.
        low_res_patch_size: Size of low-resolution patches.
        stride: Step size for patch extraction.
        colormap: Matplotlib colormap for saving images.
    
    Returns:
        Tuple of (num_high_res_patches, num_low_res_patches).
    """
    # Extract data arrays
    high_res_data = ds_high_res['analysed_sst'].values
    
    # Handle different dimension orders
    if len(ds_low_res['analysed_sst'].dims) == 3:
        low_res_data = ds_low_res['analysed_sst'][0, :, :].values
    else:
        low_res_data = ds_low_res['analysed_sst'].values
    
    # Create output directories
    hr_dir = Path(output_dir) / 'high_res' / date_str
    lr_dir = Path(output_dir) / 'low_res' / date_str
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    
    scale_factor = high_res_patch_size // low_res_patch_size
    patch_count = 0
    
    for i in range(0, high_res_data.shape[0] - high_res_patch_size, stride):
        for j in range(0, high_res_data.shape[1] - high_res_patch_size, stride):
            # Extract patches
            hr_patch = high_res_data[i:i + high_res_patch_size,
                                     j:j + high_res_patch_size]
            
            lr_i = i // scale_factor
            lr_j = j // scale_factor
            lr_patch = low_res_data[lr_i:lr_i + low_res_patch_size,
                                    lr_j:lr_j + low_res_patch_size]
            
            # Only save valid patches (no NaN values)
            if np.isnan(hr_patch).sum() == 0 and np.isnan(lr_patch).sum() == 0:
                hr_file = hr_dir / f'patch_{i}_{j}.png'
                lr_file = lr_dir / f'patch_{i}_{j}.png'
                
                plt.imsave(str(hr_file), hr_patch, cmap=colormap)
                plt.imsave(str(lr_file), lr_patch, cmap=colormap)
                
                patch_count += 1
    
    return patch_count, patch_count


def save_patches_numpy(
    ds_high_res: xr.Dataset,
    ds_low_res: xr.Dataset,
    output_dir: str,
    date_str: str,
    high_res_patch_size: int = 200,
    low_res_patch_size: int = 40,
    stride: int = 200
) -> Tuple[int, int]:
    """
    Extract and save patches as numpy arrays.
    
    This preserves the original temperature values without colormap conversion.
    
    Args:
        ds_high_res: High-resolution xarray Dataset.
        ds_low_res: Low-resolution xarray Dataset.
        output_dir: Base directory for output patches.
        date_str: Date string for directory naming.
        high_res_patch_size: Size of high-resolution patches.
        low_res_patch_size: Size of low-resolution patches.
        stride: Step size for patch extraction.
    
    Returns:
        Tuple of (num_high_res_patches, num_low_res_patches).
    """
    # Extract data arrays
    high_res_data = ds_high_res['analysed_sst'].values
    
    if len(ds_low_res['analysed_sst'].dims) == 3:
        low_res_data = ds_low_res['analysed_sst'][0, :, :].values
    else:
        low_res_data = ds_low_res['analysed_sst'].values
    
    # Create output directories
    hr_dir = Path(output_dir) / 'high_res_npy' / date_str
    lr_dir = Path(output_dir) / 'low_res_npy' / date_str
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    
    scale_factor = high_res_patch_size // low_res_patch_size
    patch_count = 0
    
    for i in range(0, high_res_data.shape[0] - high_res_patch_size, stride):
        for j in range(0, high_res_data.shape[1] - high_res_patch_size, stride):
            hr_patch = high_res_data[i:i + high_res_patch_size,
                                     j:j + high_res_patch_size]
            
            lr_i = i // scale_factor
            lr_j = j // scale_factor
            lr_patch = low_res_data[lr_i:lr_i + low_res_patch_size,
                                    lr_j:lr_j + low_res_patch_size]
            
            if np.isnan(hr_patch).sum() == 0 and np.isnan(lr_patch).sum() == 0:
                np.save(hr_dir / f'patch_{i}_{j}.npy', hr_patch)
                np.save(lr_dir / f'patch_{i}_{j}.npy', lr_patch)
                patch_count += 1
    
    return patch_count, patch_count


def visualize_patch_locations(
    ds_high_res: xr.Dataset,
    output_file: str,
    high_res_patch_size: int = 200,
    stride: int = 200
) -> int:
    """
    Create a visualization showing where patches are extracted from.
    
    Args:
        ds_high_res: High-resolution xarray Dataset.
        output_file: Path to save the visualization.
        high_res_patch_size: Size of patches.
        stride: Step size for patch extraction.
    
    Returns:
        Number of valid patch locations.
    """
    import matplotlib.patches as mpatches
    
    high_res_data = ds_high_res['analysed_sst'].values
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(high_res_data, cmap='viridis', origin='lower')
    
    patch_count = 0
    
    for i in range(0, high_res_data.shape[0] - high_res_patch_size, stride):
        for j in range(0, high_res_data.shape[1] - high_res_patch_size, stride):
            hr_patch = high_res_data[i:i + high_res_patch_size,
                                     j:j + high_res_patch_size]
            
            if np.isnan(hr_patch).sum() == 0:
                rect = mpatches.Rectangle(
                    (j, i), high_res_patch_size, high_res_patch_size,
                    linewidth=0.5, edgecolor='red', facecolor='none', alpha=0.5
                )
                ax.add_patch(rect)
                patch_count += 1
    
    ax.set_title(f'Patch Extraction Locations ({patch_count} patches)')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    
    return patch_count


def process_date_range(
    high_res_dir: str,
    low_res_dir: str,
    output_dir: str,
    start_date: str,
    end_date: str,
    high_res_pattern: str = "{date}/coarsened.nc",
    low_res_pattern: str = "{date}090000-JPL-L4_GHRSST-SSTfnd-MUR25-GLOB-v02.0-fv04.2.nc",
    save_as_numpy: bool = False
) -> None:
    """
    Process and extract patches for a range of dates.
    
    Args:
        high_res_dir: Directory containing coarsened high-res data.
        low_res_dir: Directory containing low-res data.
        output_dir: Directory to save extracted patches.
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.
        high_res_pattern: Pattern for high-res filenames.
        low_res_pattern: Pattern for low-res filenames.
        save_as_numpy: If True, save as numpy arrays; else as PNG images.
    """
    dates = pd.date_range(start_date, end_date)
    
    total_hr_patches = 0
    total_lr_patches = 0
    
    for date in tqdm(dates, desc="Extracting patches"):
        date_str = date.strftime('%Y%m%d')
        
        hr_file = Path(high_res_dir) / high_res_pattern.format(date=date_str)
        lr_file = Path(low_res_dir) / low_res_pattern.format(date=date_str)
        
        if not hr_file.exists():
            print(f"Warning: High-res file not found for {date_str}")
            continue
        
        if not lr_file.exists():
            print(f"Warning: Low-res file not found for {date_str}")
            continue
        
        try:
            with xr.open_dataset(hr_file) as ds_hr:
                with xr.open_dataset(lr_file) as ds_lr:
                    if save_as_numpy:
                        hr_count, lr_count = save_patches_numpy(
                            ds_hr, ds_lr, output_dir, date_str
                        )
                    else:
                        hr_count, lr_count = save_patches(
                            ds_hr, ds_lr, output_dir, date_str
                        )
                    
                    total_hr_patches += hr_count
                    total_lr_patches += lr_count
                    print(f'{date_str}: Extracted {hr_count} patch pairs')
        
        except Exception as e:
            print(f"Error processing {date_str}: {e}")
    
    print(f"\nTotal patches extracted: {total_hr_patches} pairs")


def main():
    """Command-line interface for patch extraction."""
    parser = argparse.ArgumentParser(
        description='Extract SST patches for super-resolution training'
    )
    parser.add_argument(
        '--high-res-dir', '-hr',
        required=True,
        help='Directory containing coarsened high-res data'
    )
    parser.add_argument(
        '--low-res-dir', '-lr',
        required=True,
        help='Directory containing low-res data'
    )
    parser.add_argument(
        '--output-dir', '-o',
        required=True,
        help='Directory to save extracted patches'
    )
    parser.add_argument(
        '--start-date', '-s',
        required=True,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date', '-e',
        required=True,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--numpy',
        action='store_true',
        help='Save patches as numpy arrays instead of PNG images'
    )
    
    args = parser.parse_args()
    
    process_date_range(
        high_res_dir=args.high_res_dir,
        low_res_dir=args.low_res_dir,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        save_as_numpy=args.numpy
    )
    
    return 0


if __name__ == '__main__':
    exit(main())
