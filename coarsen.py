"""
Data coarsening utilities for SST data preprocessing.

This module provides functions to coarsen high-resolution (0.01°) SST data
to intermediate resolution (0.05°) for use as training targets.

The coarsening is performed by selecting every 5th grid point along both
latitude and longitude dimensions.
"""

import os
import argparse
from pathlib import Path

import xarray as xr
import pandas as pd
from tqdm import tqdm


def coarsen_dataset(
    ds: xr.Dataset,
    factor: int = 5,
    time_index: int = 0
) -> xr.Dataset:
    """
    Coarsen a dataset by selecting every Nth point.
    
    Args:
        ds: Input xarray Dataset with SST data.
        factor: Coarsening factor (select every Nth point).
        time_index: Time index to select (default: 0).
    
    Returns:
        Coarsened xarray Dataset.
    """
    # Select the time index and subsample spatially
    coarsened = ds.isel(
        time=time_index,
        lat=slice(None, None, factor),
        lon=slice(None, None, factor)
    )
    return coarsened


def save_coarsened(
    input_file: str,
    output_dir: str,
    date_str: str,
    factor: int = 5
) -> None:
    """
    Load, coarsen, and save SST data for a single file.
    
    Args:
        input_file: Path to input NetCDF file.
        output_dir: Directory to save coarsened output.
        date_str: Date string for directory naming.
        factor: Coarsening factor.
    """
    # Create output directory
    output_path = Path(output_dir) / date_str
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load, coarsen, and save
    with xr.open_dataset(input_file) as ds:
        coarsened = coarsen_dataset(ds, factor=factor)
        output_file = output_path / "coarsened.nc"
        coarsened.to_netcdf(str(output_file))


def process_date_range(
    input_dir: str,
    output_dir: str,
    start_date: str,
    end_date: str,
    filename_pattern: str = "{date}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc",
    factor: int = 5
) -> None:
    """
    Process and coarsen SST data for a range of dates.
    
    Args:
        input_dir: Directory containing input NetCDF files.
        output_dir: Directory to save coarsened outputs.
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.
        filename_pattern: Pattern for input filenames with {date} placeholder.
        factor: Coarsening factor.
    """
    dates = pd.date_range(start_date, end_date)
    
    for date in tqdm(dates, desc="Coarsening data"):
        date_str = date.strftime('%Y%m%d')
        input_file = Path(input_dir) / filename_pattern.format(date=date_str)
        
        if not input_file.exists():
            print(f"Warning: Input file not found for {date_str}, skipping.")
            continue
        
        try:
            save_coarsened(
                input_file=str(input_file),
                output_dir=output_dir,
                date_str=date_str,
                factor=factor
            )
        except Exception as e:
            print(f"Error processing {date_str}: {e}")


def main():
    """Command-line interface for data coarsening."""
    parser = argparse.ArgumentParser(
        description='Coarsen high-resolution SST data'
    )
    parser.add_argument(
        '--input-dir', '-i',
        required=True,
        help='Directory containing input NetCDF files'
    )
    parser.add_argument(
        '--output-dir', '-o',
        required=True,
        help='Directory to save coarsened outputs'
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
        '--factor', '-f',
        type=int,
        default=5,
        help='Coarsening factor (default: 5)'
    )
    parser.add_argument(
        '--pattern',
        default="{date}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc",
        help='Input filename pattern with {date} placeholder'
    )
    
    args = parser.parse_args()
    
    process_date_range(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        filename_pattern=args.pattern,
        factor=args.factor
    )
    
    print("Coarsening complete!")
    return 0


if __name__ == '__main__':
    exit(main())
