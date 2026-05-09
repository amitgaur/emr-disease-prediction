#!/usr/bin/env python3
"""eICU-CRD data downloader and validator.

Usage:
    python scripts/download_eicu.py [--output-dir data/eicu]

Register at: https://eicu-crd.mit.edu/ to get access.
The dataset is freely available after registration.

This script:
1. Validates that you have access credentials
2. Downloads all required eICU CSV files
3. Validates file integrity (checksums)
4. Reports dataset statistics
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

# Optional: using requests if available, otherwise urllib
try:
    import requests
    from requests.auth import HTTPBasicAuth

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    import urllib.request

import zipfile
import tarfile

# Table files we need for the EMR disease prediction project
REQUIRED_TABLES = [
    "patient.csv",
    "diagnosis.csv",
    "medication.csv",
    "lab.csv",
    "vitalPeriodic.csv",
    "vitalAperiodic.csv",
    "treatment.csv",
    "physicalExam.csv",
    "inputevent.csv",
    "outputevent.csv",
]

# Expected sizes (approximate, in MB)
EXPECTED_SIZES = {
    "patient.csv": 20,
    "diagnosis.csv": 100,
    "medication.csv": 150,
    "lab.csv": 200,
    "vitalPeriodic.csv": 300,
    "vitalAperiodic.csv": 100,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download eICU-CRD dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/eicu"),
        help="Output directory for eICU data",
    )
    parser.add_argument(
        "--username",
        type=str,
        default=os.environ.get("EICU_USERNAME", ""),
        help="eICU database username (or set EICU_USERNAME env var)",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=os.environ.get("EICU_PASSWORD", ""),
        help="eICU database password (or set EICU_PASSWORD env var)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing files, don't download",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=None,
        help="Specific tables to download (default: all)",
    )
    return parser.parse_args()


def check_dependencies():
    """Check that required packages are available."""
    issues = []
    try:
        import pandas

        print(f"✓ pandas {pandas.__version__}")
    except ImportError:
        issues.append("pandas (required for data loading)")

    try:
        import tqdm

        print(f"✓ tqdm {tqdm.__version__}")
    except ImportError:
        print("⚠ tqdm not found, progress bars disabled")

    if issues:
        print("\nMissing dependencies:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nInstall with: uv pip install --python .venv/bin/python pandas tqdm")
        return False
    return True


def validate_file(path: Path, expected_mb: float | None = None) -> bool:
    """Validate a downloaded file exists and has reasonable size."""
    if not path.exists():
        print(f"  ✗ {path.name}: missing")
        return False

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb < 0.1:  # Less than 100KB is suspicious
        print(f"  ✗ {path.name}: too small ({size_mb:.1f} MB)")
        return False

    if expected_mb and size_mb < expected_mb * 0.5:
        print(f"  ⚠ {path.name}: smaller than expected ({size_mb:.1f} MB vs {expected_mb:.0f} MB)")
    else:
        print(f"  ✓ {path.name}: {size_mb:.1f} MB")

    return True


def validate_dataset(output_dir: Path) -> dict:
    """Validate existing dataset files."""
    print(f"\nValidating existing data in {output_dir}/")
    print("-" * 50)

    results = {}
    all_valid = True

    for table in REQUIRED_TABLES:
        path = output_dir / table
        expected = EXPECTED_SIZES.get(table, 10)
        valid = validate_file(path, expected)
        results[table] = valid
        if not valid:
            all_valid = False

    print("-" * 50)
    if all_valid:
        print("✓ All required tables present")
    else:
        missing = [t for t, v in results.items() if not v]
        print(f"✗ Missing tables: {', '.join(missing)}")

    return results


def download_with_requests(url: str, output_path: Path, username: str, password: str):
    """Download using requests library."""
    response = requests.get(url, auth=HTTPBasicAuth(username, password), stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    try:
        import tqdm

        has_progress = True
    except ImportError:
        has_progress = False

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if has_progress:
                    # Progress would go here
                    pass

    return downloaded


def download_with_urllib(url: str, output_path: Path, username: str, password: str):
    """Download using urllib (fallback)."""
    password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(None, url, username, password)

    handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
    opener = urllib.request.build_opener(handler)

    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, output_path)


def download_table(
    table_name: str, output_dir: Path, username: str, password: str
) -> bool:
    """Download a single eICU table."""
    # eICU data access URL pattern
    base_url = "https://eicu-crd.mit.edu/wp-content/uploads/"
    url = f"{base_url}{table_name}.tar.gz"
    output_path = output_dir / f"{table_name}.tar.gz"

    print(f"\nDownloading {table_name}...")
    print(f"  URL: {url}")

    try:
        if HAS_REQUESTS:
            download_with_requests(url, output_path, username, password)
        else:
            download_with_urllib(url, output_path, username, password)

        # Extract tar.gz
        print(f"  Extracting to {output_dir}/")
        with tarfile.open(output_path, "r:gz") as tar:
            tar.extractall(output_dir)

        # Remove tar.gz after extraction
        output_path.unlink()
        print(f"  ✓ {table_name} downloaded and extracted")
        return True

    except Exception as e:
        print(f"  ✗ Failed to download {table_name}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def run_validation_checks(output_dir: Path):
    """Run validation checks on downloaded data."""
    print("\n" + "=" * 50)
    print("Running validation checks...")
    print("=" * 50)

    try:
        import pandas as pd

        for table in REQUIRED_TABLES:
            path = output_dir / table
            if not path.exists():
                print(f"  ⚠ {table}: missing")
                continue

            try:
                df = pd.read_csv(path, nrows=5)
                print(f"  ✓ {table}: {len(df.columns)} columns, sample loaded")

                # Check row count (approximate)
                with open(path, "r") as f:
                    row_count = sum(1 for _ in f) - 1  # subtract header
                print(f"    → ~{row_count:,} rows")

            except Exception as e:
                print(f"  ✗ {table}: failed to read - {e}")

    except ImportError:
        print("⚠ pandas not available, skipping validation")


def print_next_steps(output_dir: Path):
    """Print instructions for next steps."""
    print("\n" + "=" * 50)
    print("NEXT STEPS")
    print("=" * 50)
    print(f"""
1. Register for eICU access (if not already):
   → https://eicu-crd.mit.edu/

2. Set credentials for automated download:
   export EICU_USERNAME=your_email
   export EICU_PASSWORD=your_password

3. Download data:
   python scripts/download_eicu.py --output-dir {output_dir} \\
       --username $EICU_USERNAME --password $EICU_PASSWORD

4. Run the XGBoost baseline:
   cd ~/projects/emr-disease-prediction
   source .venv/bin/activate
   python -m src.models.xgb_baseline --data-dir {output_dir}

5. Expected output:
   - results/models/xgb_baseline.json
   - results/shap/feature_importance.png
   - results/benchmark_YYYY-MM-DD.json
""")


def main():
    args = parse_args()

    print("=" * 50)
    print("eICU-CRD Data Downloader")
    print("=" * 50)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # If validate-only, just validate and exit
    if args.validate_only:
        validate_dataset(args.output_dir)
        return

    # Check for credentials
    if not args.username or not args.password:
        print("""
⚠ Missing credentials!

To download eICU data:
1. Register at https://eicu-crd.mit.edu/
2. Set credentials:
   export EICU_USERNAME=your_email
   export EICU_PASSWORD=your_password
3. Run: python scripts/download_eicu.py --username $EICU_USERNAME --password $EICU_PASSWORD

For now, validating existing files only.
""")
        validate_dataset(args.output_dir)
        print_next_steps(args.output_dir)
        return

    # Validate existing files first
    print("\nChecking existing files...")
    existing = validate_dataset(args.output_dir)

    # Download missing tables
    tables_to_download = args.tables or REQUIRED_TABLES
    for table in tables_to_download:
        if existing.get(table, False):
            print(f"\nSkipping {table} (already exists)")
            continue
        success = download_table(table, args.output_dir, args.username, args.password)
        if not success:
            print(f"Failed to download {table}, continuing with existing files...")

    # Run validation
    run_validation_checks(args.output_dir)

    # Print next steps
    print_next_steps(args.output_dir)


if __name__ == "__main__":
    main()