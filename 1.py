"""
3GPP Specification Downloader
Downloads the latest versions of specified 3GPP specs as ZIP files
from the official 3GPP FTP server (https://www.3gpp.org).

Requirements: pip install requests
Usage:        python download_3gpp_specs.py
"""

import os
import re
import requests

# Output folder
OUTPUT_DIR = "3gpp_specs"

# Specs to download: (3GPP number, series, friendly label)
SPECS = [
    ("38.300", "38", "NR + NG-RAN overall description"),
    ("38.401", "38", "NG-RAN architecture"),
    ("38.331", "38", "RRC protocol"),
    ("38.211", "38", "Physical channels & modulation"),
    ("38.213", "38", "Physical layer procedures (control)"),
    ("38.214", "38", "Physical layer procedures (data)"),
    ("23.501", "23", "5G system architecture"),
    ("23.502", "23", "5G procedures"),
]

BASE_FTP = "https://www.3gpp.org/ftp/Specs/archive"


def get_latest_zip_url(spec_number: str, series: str) -> str | None:
    """
    Browse the 3GPP FTP index page for a spec and return the URL of
    the highest-versioned ZIP file found.
    """
    # e.g. 38.300 -> folder name 38300
    folder = spec_number.replace(".", "")
    index_url = f"{BASE_FTP}/{series}_series/{folder}/"

    try:
        resp = requests.get(index_url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  [ERROR] Could not fetch index for {spec_number}: {exc}")
        return None

    # Find all zip filenames in the HTML listing
    zips = re.findall(rf'{folder}-[\w.]+\.zip', resp.text, flags=re.IGNORECASE)
    if not zips:
        print(f"  [WARN]  No ZIP files found in index for {spec_number}")
        return None

    # Sort by version number embedded in filename and pick the latest
    zips_sorted = sorted(set(zips))
    latest = zips_sorted[-1]
    return f"{index_url}{latest}"


def download_file(url: str, dest_path: str) -> bool:
    """Stream-download a file to dest_path, showing progress."""
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
            size_kb = downloaded // 1024
            print(f"  [OK]    Saved {os.path.basename(dest_path)} ({size_kb:,} KB)")
            return True
    except requests.RequestException as exc:
        print(f"  [ERROR] Download failed: {exc}")
        return False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving specs to: {os.path.abspath(OUTPUT_DIR)}\n")

    results = []
    for spec_num, series, label in SPECS:
        print(f"→ TS {spec_num}  ({label})")

        url = get_latest_zip_url(spec_num, series)
        if url is None:
            results.append((spec_num, False))
            continue

        filename = url.split("/")[-1]
        dest = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(dest):
            print(f"  [SKIP]  Already downloaded: {filename}")
            results.append((spec_num, True))
            continue

        print(f"  Downloading: {url}")
        ok = download_file(url, dest)
        results.append((spec_num, ok))
        print()

    # Summary
    print("\n=== Summary ===")
    for spec_num, ok in results:
        status = "✓" if ok else "✗"
        print(f"  {status}  TS {spec_num}")


if __name__ == "__main__":
    main()