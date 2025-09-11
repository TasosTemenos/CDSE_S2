#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from osgeo import gdal  # kept as requested

from collections import defaultdict

import numpy as np
import xarray as xr
import rioxarray as rxr
from rasterio.enums import Resampling

# -------- configuration --------
BANDS_10M = ["B02", "B03", "B04", "B08"]
BANDS_20M = ["B8A", "B11", "B12"]
ALL_BANDS = BANDS_10M + BANDS_20M
SCL_MASK_VALUES = {0, 1, 3, 8, 9, 10, 11}  # invalids, cloud/shadow, cirrus, snow/ice
SEASON_ORDER = ("M1", "M2", "M3", "M4")

# spatial chunking (tune for your RAM)
CHUNK_X = 1024
CHUNK_Y = 1024

# default nodata for uint16 output
NODATA_U16 = 0


# -------- helpers (find files) --------
def find_band_files(root: Path, band: str) -> List[Path]:
    """Find all JP2 files for a band under root (both '10m'/'20m' and uppercase variants)."""
    res = "10m" if band in BANDS_10M else "20m"
    pats = [f"**/*_{band}_{res}.jp2", f"**/*_{band}_{res.upper()}.jp2"]
    out: List[Path] = []
    for p in pats:
        out += list(root.glob(p))
    return sorted(out)


def find_granule_dir(path: Path) -> Optional[Path]:
    """Climb up to find the GRANULE/L2A_* folder for a given band path."""
    cur = path
    for _ in range(12):
        if cur.parent and cur.parent.name == "GRANULE" and cur.name.startswith("L2A_"):
            return cur
        cur = cur.parent
    return None


def granule_key(p: Path) -> Optional[str]:
    """Stable string key for a granule directory."""
    g = find_granule_dir(p)
    return None if g is None else str(g.resolve())


def find_scl_in_granule(granule_path: Path) -> Optional[Path]:
    """Find the 20m SCL JP2 inside a granule directory."""
    for rel in ("IMG_DATA/R20m/*SCL*20m*.jp2", "QI_DATA/*SCL*20m*.jp2"):
        m = list(granule_path.glob(rel))
        if m:
            m.sort()
            return m[0]
    return None


# -------- IO / reprojection --------
def open_ref_da(b02_path: Path) -> xr.DataArray:
    """Open reference B02 as a single-band DataArray with spatial chunks."""
    da = rxr.open_rasterio(
        b02_path,
        chunks={"band": 1, "x": CHUNK_X, "y": CHUNK_Y}
    ).squeeze("band", drop=True)
    if da.rio.crs is None:
        raise ValueError(f"No CRS found in {b02_path}")
    return da.chunk({"x": CHUNK_X, "y": CHUNK_Y})


def reproject_to_ref(path: Path, ref: xr.DataArray, resampling: Resampling) -> xr.DataArray:
    """Open a file and reproject/align to the reference grid."""
    da = rxr.open_rasterio(
        path,
        chunks={"band": 1, "x": CHUNK_X, "y": CHUNK_Y}
    ).squeeze("band", drop=True)
    da = da.rio.reproject_match(ref, resampling=resampling)
    return da.chunk({"x": CHUNK_X, "y": CHUNK_Y})


# -------- inventory (optimization #2) --------
def build_scene_inventory(season_safe_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Build a per-season inventory of available scenes (granules) and their band file paths.
    Returns:
        inv: { granule_key: {"B02": Path, "B03": Path, ..., "SCL": Path|None} }
    """
    inv: Dict[str, Dict[str, Path]] = defaultdict(dict)

    # Collect band files per granule
    for b in ALL_BANDS:
        for p in find_band_files(season_safe_dir, b):
            gk = granule_key(p)
            if gk is None:
                continue
            inv[gk][b] = p

    # Collect SCL once per granule
    for gk in list(inv.keys()):
        gpath = Path(gk)
        scl = find_scl_in_granule(gpath)
        if scl is not None:
            inv[gk]["SCL"] = scl

    return inv


# -------- processing --------
def build_season_median(
    season_safe_dir: Path,
    out_tif: Path,
    nodata_val: int
) -> None:
    """
    Build a 7-band (B02,B03,B04,B08,B8A,B11,B12) seasonal median mosaic at 10 m,
    masked by SCL invalid classes, and write as uint16 with nodata.
    """
    # Build inventory once (scenes -> band paths + SCL)
    inv = build_scene_inventory(season_safe_dir)
    if not inv:
        print(f"[WARN] No scenes found in {season_safe_dir}; skipping.")
        return

    # Choose reference grid from the first scene that has B02
    ref_b02_path: Optional[Path] = None
    for gk, items in inv.items():
        if "B02" in items:
            ref_b02_path = items["B02"]
            break
    if ref_b02_path is None:
        print(f"[WARN] No B02 JP2 files found in {season_safe_dir}; skipping.")
        return
    ref = open_ref_da(ref_b02_path)

    # Per-granule SCL cache (optimization #1)
    scl_cache: Dict[str, Optional[xr.DataArray]] = {}

    # Collect per-band stacks (each element is one scene)
    band_stacks: Dict[str, List[xr.DataArray]] = {b: [] for b in ALL_BANDS}
    scl_mask_values_list = list(SCL_MASK_VALUES)  # avoid recreating each time

    # Iterate scenes; for each band present, reproject & mask (reuse SCL per scene)
    for gk, items in inv.items():
        # Prepare SCL mask once per scene (if SCL exists)
        scl_mask = None
        if "SCL" in items:
            if gk not in scl_cache:
                try:
                    scl_reproj = reproject_to_ref(items["SCL"], ref, Resampling.nearest)
                except Exception as e:
                    print(f"[INFO] SCL reprojection failed for scene {gk}: {e}")
                    scl_reproj = None
                scl_cache[gk] = scl_reproj
            scl_reproj = scl_cache[gk]
            if scl_reproj is not None:
                try:
                    scl_mask = scl_reproj.isin(scl_mask_values_list)  # True where cloudy/invalid
                except Exception as e:
                    print(f"[INFO] SCL mask creation failed for scene {gk}: {e}")
                    scl_mask = None

        # Reproject/append each band available in this scene
        for band in ALL_BANDS:
            if band not in items:
                continue
            resamp = Resampling.bilinear if band in BANDS_20M else Resampling.nearest
            p = items[band]
            try:
                da = reproject_to_ref(p, ref, resamp).astype("float32")
                if scl_mask is not None:
                    da = da.where(~scl_mask)
                band_stacks[band].append(da)
            except Exception as e:
                print(f"[SKIP] {Path(p).name}: {e}")

        # free per-scene mask early
        del scl_mask

    # Compute per-band median; store in dict
    med_map: Dict[str, xr.DataArray] = {}
    for b, lst in band_stacks.items():
        if not lst:
            continue
        stack = xr.concat(lst, dim="time").chunk({"time": -1, "x": CHUNK_X, "y": CHUNK_Y})
        med = stack.quantile(q=0.5, dim="time", skipna=True)  # scalar quantile (no 'quantile' dim)
        med_map[b] = med
        del stack

    if not med_map:
        print(f"[WARN] No valid data after masking in {season_safe_dir}; skipping.")
        return

    # enforce EXACT 7 bands and correct order
    ordered = []
    for b in ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]:
        if b in med_map:
            ordered.append(med_map[b])
        else:
            ordered.append(xr.full_like(ref, np.nan, dtype="float32"))

    out = xr.concat(ordered, dim="band").assign_coords({"band": np.arange(1, 8)})
    out = out.rio.write_crs(ref.rio.crs).rio.write_transform(ref.rio.transform())

    # ---- write as uint16 with nodata ----
    out_u16 = (
        out.fillna(nodata_val)
           .round()
           .clip(min=0, max=65535)
           .astype("uint16")
           .rio.write_nodata(nodata_val)
           .chunk({"band": -1, "x": CHUNK_X, "y": CHUNK_Y})
    )

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    out_u16.rio.to_raster(
        out_tif,
        dtype="uint16",
        tiled=True,
        compress="deflate",  # change to "zstd" if your GDAL supports it
        predictor=2,
        BIGTIFF="IF_SAFER",
    )
    print(f"[OK] Saved {out_tif} (uint16; bands: B02,B03,B04,B08,B8A,B11,B12; nodata={nodata_val})")


# -------- multi-tile utils --------
def read_tiles_from_txt(path: Path) -> List[str]:
    """
    Read a text file and extract tile codes.
    Accepts one per line and/or comma-separated, ignores blanks, de-dupes preserving order.
    """
    txt = path.read_text(encoding="utf-8")
    raw = [t.strip() for t in txt.replace(",", "\n").splitlines()]
    tiles = [t for t in raw if t]
    seen = set()
    ordered: List[str] = []
    for t in tiles:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def process_one_tile(root: Path, year: int, tile: str, seasons: List[str], nodata_val: int) -> None:
    for label in seasons:
        season_dir = root / f"{year}" / tile / label
        safe_dir = season_dir / "SAFE"
        out_tif = season_dir / f"mosaic_{year}_{label}_{tile}.tif"
        print(f"=== {tile} | {label} -> {safe_dir} ===")
        build_season_median(safe_dir, out_tif, nodata_val=nodata_val)


def main():
    global CHUNK_X, CHUNK_Y, NODATA_U16  # must be first in main()
    ap = argparse.ArgumentParser(
        description="Process S2 L2A JP2 .SAFE into seasonal cloud-free median mosaics (7 bands @10 m, DN uint16)."
    )
    ap.add_argument("--root", type=Path, required=True,
                    help="Root folder used by downloader (e.g., ./cdse_s2_2018)")
    ap.add_argument("--year", type=int, required=True)

    # Either a single tile or a file with many tiles
    mx = ap.add_mutually_exclusive_group(required=True)
    mx.add_argument("--tile", help="Single tile ID (e.g., T34SDJ)")
    mx.add_argument("--tiles_txt", type=Path,
                    help="Path to a text file with tile IDs (one per line or comma-separated)")

    ap.add_argument("--season", choices=SEASON_ORDER,
                    help="Process a single season (M1..M4). If omitted, processes all.")
    ap.add_argument("--chunk", type=int, default=CHUNK_X,
                    help="Spatial chunk size for x and y (default 1024).")
    ap.add_argument("--nodata", type=int, default=NODATA_U16,
                    help="Nodata value for uint16 output (0 or 65535).")

    args = ap.parse_args()

    CHUNK_X = CHUNK_Y = int(args.chunk)
    NODATA_U16 = int(args.nodata)

    seasons = [args.season] if args.season else list(SEASON_ORDER)

    # Build tile list
    if args.tile:
        tiles = [args.tile]
    else:
        tiles = read_tiles_from_txt(args.tiles_txt)
        if not tiles:
            raise SystemExit(f"No tiles found in {args.tiles_txt}")

    root = Path(args.root)
    year = int(args.year)

    for t in tiles:
        process_one_tile(root, year, t, seasons, nodata_val=NODATA_U16)


if __name__ == "__main__":
    main()
