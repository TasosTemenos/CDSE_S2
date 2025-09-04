#!/usr/bin/env python3
"""
CDSE Sentinel-2 L2A downloader + seasonal cloud-free median mosaics at 10 m

Example run (tile T34SGH, year 2018):
    python cdse_s2_seasonal.py --tile T34SGH --year 2018 --outdir ./cdse_s2_2018 --max-cloud 80

Outputs (per season):
  outdir/2018/T34SGH/M1/mosaic_2018_M1_T34SGH.tif
  outdir/2018/T34SGH/M2/mosaic_2018_M2_T34SGH.tif
  outdir/2018/T34SGH/M3/mosaic_2018_M3_T34SGH.tif
  outdir/2018/T34SGH/M4/mosaic_2018_M4_T34SGH.tif

Notes:
- Requires GDAL with JP2K (openjpeg) support (installed via conda-forge above).
- Cloud masking uses L2A SCL (classes 3,8,9,10,11 masked; also 0,1 are treated as invalid).
- B8A, B11, B12 are native 20 m and are resampled to 10 m to match B02 grid.
- Values are exported as float32 reflectance (DN/10000.0), **not** scaled by BOA_ADD_OFFSET (not needed for 2018).
"""

from __future__ import annotations
import argparse
import datetime as dt
import getpass
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from osgeo import gdal
import rasterio
from rasterio.enums import Resampling
import rioxarray as rxr
import xarray as xr

CDSE_OAUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
ODATA_BASE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
ZIPPER_BASE = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"

BANDS_10M = ["B02", "B03", "B04", "B08"]
BANDS_20M = ["B8A", "B11", "B12"]
ALL_BANDS = BANDS_10M + BANDS_20M

# SCL classes to mask (clouds, cloud shadow, cirrus, snow/ice) and invalids
SCL_MASK_VALUES = {0, 1, 3, 8, 9, 10, 11}

# ---- Auth ----

def get_access_token(username: str, password: str) -> str:
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    r = requests.post(CDSE_OAUTH_URL, data=data)
    r.raise_for_status()
    return r.json()["access_token"]

# ---- Seasons ----

def seasons_for_year(year: int) -> List[Tuple[str, dt.datetime, dt.datetime]]:
    """Return [(label, start_dt, end_dt_exclusive)] for M1..M4.
    M1 is Dec of previous year -> Mar 1 of target year.
    """
    y = year
    return [
        ("M1", dt.datetime(y - 1, 12, 1), dt.datetime(y, 3, 1)),
        ("M2", dt.datetime(y, 3, 1), dt.datetime(y, 6, 1)),
        ("M3", dt.datetime(y, 6, 1), dt.datetime(y, 9, 1)),
        ("M4", dt.datetime(y, 9, 1), dt.datetime(y, 12, 1)),
    ]

# ---- Search ----

def odata_url(tile: str, start: dt.datetime, end: dt.datetime, max_cloud: float) -> str:
    # OData filter: Sentinel-2 collection, date window, product type MSIL2A, this tile, cloud cover <= max
    # Use contains(Name,'_TxxYYY_') to select granules for the tile.
    f = (
        "Collection/Name eq 'SENTINEL-2' "
        f"and ContentDate/Start ge {start.strftime('%Y-%m-%d')}T00:00:00.000Z "
        f"and ContentDate/Start lt {end.strftime('%Y-%m-%d')}T00:00:00.000Z "
        "and contains(Name, 'MSIL2A') "
        f"and contains(Name, '_{tile}_') "
        "and not contains(Name, '_COG') "
        f"and Attributes/OData.CSC.DoubleAttribute/any(att: att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {max_cloud})"
    )
    # order by time ascending; large $top to reduce pagination; we'll still follow @odata.nextLink
    params = {
        "$filter": f,
        "$orderby": "ContentDate/Start asc",
        "$top": 200,
    }
    # Compose URL manually to preserve spaces and special tokens in $filter
    def enc(v: str) -> str:
        return requests.utils.requote_uri(v)
    q = "&".join([f"{k}={enc(str(v))}" for k, v in params.items()])
    return f"{ODATA_BASE}?{q}"


def search_products(tile: str, start: dt.datetime, end: dt.datetime, max_cloud: float, token: str) -> pd.DataFrame:
    url = odata_url(tile, start, end, max_cloud)
    headers = {"Authorization": f"Bearer {token}"}
    out = []
    while url:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        js = resp.json()
        out.extend(js.get("value", []))
        url = js.get("@odata.nextLink")
    if not out:
        return pd.DataFrame(columns=["Id", "Name", "ContentDate"])
    df = pd.DataFrame(out)
    # Keep only essential fields
    keep_cols = [c for c in ["Id", "Name", "ContentDate"] if c in df.columns]
    return df[keep_cols].copy()

# ---- Download ----

def download_product(product_id: str, dest_zip: Path, token: str, retries: int = 3) -> None:
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    url = f"{ZIPPER_BASE}({product_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                with open(dest_zip, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest_zip.name) as pbar:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            # rudimentary integrity check: ensure non-zero length
            if dest_zip.stat().st_size < 1024 * 1024:
                raise IOError("Downloaded file is unexpectedly small.")
            return
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(2 * attempt)


def extract_safe(zip_path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        # Find .SAFE root folder inside archive
        safe_root = None
        for name in z.namelist():
            if name.endswith(".SAFE/") and "/" not in name[:-6]:
                safe_root = name.rstrip("/")
                break
        z.extractall(dest_dir)
        # Return path to extracted .SAFE dir
        if safe_root is None:
            # Fallback: try to detect .SAFE
            roots = [n for n in z.namelist() if n.endswith(".SAFE/")]
            if roots:
                safe_root = roots[0].rstrip("/")
        extracted = dest_dir / (safe_root or zip_path.stem)
        return extracted

# ---- File discovery ----

def find_band_file(safe_dir: Path, band: str) -> Path:
    # band like 'B02' or 'B8A'; handle 10m/20m suffix in pattern
    res = "10m" if band in BANDS_10M else "20m"
    patterns = [
        f"**/*_{band}_{res}.*",
        f"**/*_{band}_{res.upper()}.*",
    ]
    for pat in patterns:
        matches = list(safe_dir.glob(pat))
        if matches:
            # Prefer JP2 if multiple
            matches.sort(key=lambda p: (p.suffix.lower() != ".jp2", len(str(p))))
            return matches[0]
    raise FileNotFoundError(f"Cannot locate {band} ({res}) in {safe_dir}")


def find_scl_file(safe_dir: Path) -> Path:
    # SCL typically in IMG_DATA/R20m or QI_DATA; accept JP2/TCI variants
    for pat in ("**/*SCL_20m.*", "**/QI_DATA/*SCL*20m*", "**/IMG_DATA/R20m/*SCL*20m*"):
        matches = list(safe_dir.glob(pat))
        if matches:
            matches.sort(key=lambda p: (p.suffix.lower() != ".jp2", len(str(p))))
            return matches[0]
    raise FileNotFoundError(f"Cannot locate SCL_20m in {safe_dir}")

# ---- Mosaic building ----

def open_ref_da(b02_path: Path) -> xr.DataArray:
    da = rxr.open_rasterio(b02_path, chunks={"band": 1, "x": 2048, "y": 2048}).squeeze("band", drop=True)
    # ensure CRS & transform are attached
    da.rio.write_crs(da.rio.crs, inplace=True)
    return da


def reproject_to_ref(path: Path, ref: xr.DataArray, resampling: Resampling) -> xr.DataArray:
    da = rxr.open_rasterio(path, chunks={"band": 1, "x": 2048, "y": 2048}).squeeze("band", drop=True)
    da = da.rio.reproject_match(ref, resampling=resampling)
    return da


def build_season_median(safe_dirs: List[Path], out_tif: Path) -> None:
    if not safe_dirs:
        print(f"No scenes for season; skipping {out_tif}")
        return

    # Use B02 from the first scene as reference grid
    ref_b02 = find_band_file(safe_dirs[0], "B02")
    ref_da = open_ref_da(ref_b02)

    # Collect per-band stacks
    band_stacks: Dict[str, List[xr.DataArray]] = {b: [] for b in ALL_BANDS}

    for sd in safe_dirs:
        try:
            scl_path = find_scl_file(sd)
            scl = reproject_to_ref(scl_path, ref_da, Resampling.nearest)
            # mask True where SCL indicates cloud/shadow/cirrus/snow or invalid
            mask = xr.apply_ufunc(np.isin, scl, xr.DataArray(np.array(list(SCL_MASK_VALUES), dtype=np.uint8), dims=["classes"]))
            mask = mask.any("classes")
            # Read each band, reproject if needed, apply mask and scale to reflectance
            for b in ALL_BANDS:
                bpath = find_band_file(sd, b)
                resamp = Resampling.bilinear if b in BANDS_20M else Resampling.nearest
                da = reproject_to_ref(bpath, ref_da, resamp)
                # Convert to float reflectance (DN/10000) and mask
                da = da.astype("float32") / 10000.0
                da = da.where(~mask)
                band_stacks[b].append(da)
        except Exception as e:
            print(f"Warning: skipping scene {sd.name}: {e}")
            continue

    # Compute median per band (lazy with dask)
    medians = []
    band_names = []
    for b in ALL_BANDS:
        lst = band_stacks[b]
        if not lst:
            print(f"No valid data for band {b}; skipping")
            continue
        stack = xr.concat(lst, dim="time")
        med = stack.median(dim="time", skipna=True)
        medians.append(med)
        band_names.append(b)

    if not medians:
        print(f"No valid data for any band; skipping {out_tif}")
        return

    # Stack into one 3D array (band, y, x)
    out = xr.concat(medians, dim="band")
    out = out.assign_coords({"band": xr.DataArray(np.arange(1, len(medians) + 1), dims=("band",))})

    # Ensure spatial metadata from ref
    out = out.rio.write_crs(ref_da.rio.crs)
    out = out.rio.write_transform(ref_da.rio.transform())

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    # Save
    compress = {"compress": "deflate"}
    out.rio.to_raster(out_tif, dtype="float32", tiled=True, **compress)
    print(f"Saved {out_tif} (bands: {band_names})")

# ---- Orchestration ----

def run(tile: str, year: int, outdir: Path, username: str | None, password: str | None, max_cloud: float) -> None:
    if username is None:
        username = input("CDSE username (email): ")
    if password is None:
        password = getpass.getpass("CDSE password: ")

    token = get_access_token(username, password)

    for label, start, end in seasons_for_year(year):
        print(f"=== {label}: {start.date()} -> {end.date()} ===")
        df = search_products(tile, start, end, max_cloud, token)
        if df.empty:
            print(f"No products found for {label}")
            continue

        season_dir = outdir / f"{year}" / tile / label
        raw_dir = season_dir / "raw"
        safe_dirs: List[Path] = []

        for _, row in df.iterrows():
            pid = row["Id"]
            name = row["Name"]
            zip_path = raw_dir / f"{name}.zip"
            safe_outdir = season_dir / "SAFE"

            if not zip_path.exists():
                print(f"Downloading {name} ...")
                download_product(pid, zip_path, token)
            else:
                print(f"Exists: {zip_path.name}")

            # Extract
            try:
                extracted = extract_safe(zip_path, safe_outdir)
                safe_dirs.append(extracted)
            except Exception as e:
                print(f"Failed to extract {zip_path.name}: {e}")

        # Build seasonal mosaic
        out_tif = season_dir / f"mosaic_{year}_{label}_{tile}.tif"
        build_season_median(safe_dirs, out_tif)


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CDSE S2 L2A seasonal median mosaics at 10 m")
    p.add_argument("--tile", required=True, help="MGRS tile, e.g., T34SGH")
    p.add_argument("--year", type=int, required=True, help="Target year (M1 spans Dec of year-1)")
    p.add_argument("--outdir", type=Path, default=Path("./cdse_s2_output"))
    p.add_argument("--username", default=None)
    p.add_argument("--password", default=None)
    p.add_argument("--max-cloud", type=float, default=80.0, help="Max cloud cover % per product (OData prefilter)")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    run(tile=args.tile, year=args.year, outdir=args.outdir, username=args.username, password=args.password, max_cloud=args.max_cloud)