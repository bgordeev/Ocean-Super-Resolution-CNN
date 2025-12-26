"""
Data download utilities for NASA MUR SST dataset.

This module provides functions to download high-resolution (MUR v4.1) and
low-resolution (MUR25 v4.2) sea surface temperature data from NASA's
Physical Oceanography Distributed Active Archive Center (PO.DAAC).

Requires NASA Earthdata credentials for authentication.
"""

import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import requests
from tqdm import tqdm


class SessionWithHeaderRedirection(requests.Session):
    """
    Custom requests session that handles NASA Earthdata authentication.
    
    This session class properly manages the Authorization header during
    redirects, which is required for NASA Earthdata authentication flow.
    
    Attributes:
        AUTH_HOST: The NASA Earthdata authentication hostname.
    """
    
    AUTH_HOST = 'urs.earthdata.nasa.gov'

    def __init__(self, username: str, password: str):
        """
        Initialize the session with Earthdata credentials.
        
        Args:
            username: NASA Earthdata username.
            password: NASA Earthdata password.
        """
        super().__init__()
        self.auth = (username, password)

    def rebuild_auth(self, prepared_request, response):
        """
        Override to handle Authorization header on redirect.
        
        Removes the Authorization header when redirecting to a different
        host (except the AUTH_HOST) to prevent credential leakage.
        
        Args:
            prepared_request: The prepared request object.
            response: The response that triggered the redirect.
        """
        headers = prepared_request.headers
        url = prepared_request.url

        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)

            if (original_parsed.hostname != redirect_parsed.hostname and
                redirect_parsed.hostname != self.AUTH_HOST and
                original_parsed.hostname != self.AUTH_HOST):
                del headers['Authorization']


def download_file(session: requests.Session, url: str, filename: str) -> None:
    """
    Download a file from a URL using the provided session.
    
    Args:
        session: Authenticated requests session.
        url: URL of the file to download.
        filename: Local path where the file will be saved.
    
    Raises:
        requests.HTTPError: If the download fails.
    """
    with session.get(url, stream=True) as response:
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True, 
                     desc=Path(filename).name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    pbar.update(len(chunk))


def download_sst_data(
    username: str,
    password: str,
    start_date: str,
    end_date: str,
    output_dir: str = "data/raw",
    download_high_res: bool = True,
    download_low_res: bool = True
) -> None:
    """
    Download MUR SST data for a date range.
    
    Downloads both high-resolution (0.01°, MUR v4.1) and low-resolution
    (0.25°, MUR25 v4.2) sea surface temperature data from PO.DAAC.
    
    Args:
        username: NASA Earthdata username.
        password: NASA Earthdata password.
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.
        output_dir: Directory to save downloaded files.
        download_high_res: Whether to download high-resolution data.
        download_low_res: Whether to download low-resolution data.
    """
    session = SessionWithHeaderRedirection(username, password)
    
    # Create output directories
    hr_dir = Path(output_dir) / "high_res"
    lr_dir = Path(output_dir) / "low_res"
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse dates
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # URL templates
    HR_URL_TEMPLATE = (
        "https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/"
        "MUR-JPL-L4-GLOB-v4.1/{date}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc"
    )
    LR_URL_TEMPLATE = (
        "https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/"
        "MUR25-JPL-L4-GLOB-v04.2/{date}090000-JPL-L4_GHRSST-SSTfnd-MUR25-GLOB-v02.0-fv04.2.nc"
    )
    
    # Download for each date
    current = start
    while current <= end:
        date_str = current.strftime('%Y%m%d')
        
        if download_high_res:
            hr_url = HR_URL_TEMPLATE.format(date=date_str)
            hr_filename = hr_dir / f"{date_str}_high_res.nc"
            
            if not hr_filename.exists():
                print(f"\nDownloading high-res data for {date_str}...")
                try:
                    download_file(session, hr_url, str(hr_filename))
                except requests.HTTPError as e:
                    print(f"Failed to download high-res for {date_str}: {e}")
            else:
                print(f"High-res data for {date_str} already exists, skipping.")
        
        if download_low_res:
            lr_url = LR_URL_TEMPLATE.format(date=date_str)
            lr_filename = lr_dir / f"{date_str}_low_res.nc"
            
            if not lr_filename.exists():
                print(f"\nDownloading low-res data for {date_str}...")
                try:
                    download_file(session, lr_url, str(lr_filename))
                except requests.HTTPError as e:
                    print(f"Failed to download low-res for {date_str}: {e}")
            else:
                print(f"Low-res data for {date_str} already exists, skipping.")
        
        current += timedelta(days=1)
    
    print("\nDownload complete!")


def main():
    """Command-line interface for data download."""
    parser = argparse.ArgumentParser(
        description='Download MUR SST data from NASA PO.DAAC'
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
        '--output-dir', '-o',
        default='data/raw',
        help='Output directory (default: data/raw)'
    )
    parser.add_argument(
        '--username', '-u',
        default=os.environ.get('EARTHDATA_USERNAME'),
        help='NASA Earthdata username (or set EARTHDATA_USERNAME env var)'
    )
    parser.add_argument(
        '--password', '-p',
        default=os.environ.get('EARTHDATA_PASSWORD'),
        help='NASA Earthdata password (or set EARTHDATA_PASSWORD env var)'
    )
    parser.add_argument(
        '--high-res-only',
        action='store_true',
        help='Download only high-resolution data'
    )
    parser.add_argument(
        '--low-res-only',
        action='store_true',
        help='Download only low-resolution data'
    )
    
    args = parser.parse_args()
    
    if not args.username or not args.password:
        print("Error: NASA Earthdata credentials required.")
        print("Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables")
        print("or use --username and --password arguments.")
        return 1
    
    download_high = not args.low_res_only
    download_low = not args.high_res_only
    
    download_sst_data(
        username=args.username,
        password=args.password,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        download_high_res=download_high,
        download_low_res=download_low
    )
    
    return 0


if __name__ == '__main__':
    exit(main())
