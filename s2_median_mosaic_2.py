#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Optional

from collections import defaultdict

import numpy as np
import xarray as xr
import rioxarray as rxr
from rasterio.enums import Resampling

from multi_tile_down_seasons import normalize_tile, read_tile_file

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
    with rxr.open_rasterio(
        b02_path,
        chunks={"band": 1, "x": CHUNK_X, "y": CHUNK_Y}
    ) as source:
        if source.rio.crs is None:
            raise ValueError(f"No CRS found in {b02_path}")
        return source.squeeze("band", drop=True)


def reproject_to_ref(path: Path, ref: xr.DataArray, resampling: Resampling) -> xr.DataArray:
    """Open a file and reproject/align to the reference grid."""
    with rxr.open_rasterio(
        path, masked=True,
        chunks={"band": 1, "x": CHUNK_X, "y": CHUNK_Y}
    ) as source:
        da = source.squeeze("band", drop=True).rio.reproject_match(ref, resampling=resampling)
    return da.chunk({"x": CHUNK_X, "y": CHUNK_Y})


# -------- inventory (optimization #2) --------
def build_scene_inventory(safe_roots: List[Path]) -> Dict[str, Dict[str, Path]]:
    """
    Build a per-season inventory of available scenes (granules) and their band file paths.
    Returns:
        inv: { granule_key: {"B02": Path, "B03": Path, ..., "SCL": Path|None} }
    """
    inv: Dict[str, Dict[str, Path]] = defaultdict(dict)

    # Collect band files per granule
    for b in ALL_BANDS:
        for root in safe_roots:
            for p in find_band_files(root, b):
                gk = granule_key(p)
                if gk is not None:
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
    nodata_val: int,
    *,
    scaled: bool = False,
    products: Optional[List[Path]] = None,
) -> None:
    """
    Build a 7-band (B02,B03,B04,B08,B8A,B11,B12) seasonal median mosaic at 10 m,
    masked by SCL invalid classes, and write as uint16 with nodata.
    """
    # Build inventory once (scenes -> band paths + SCL)
    inv = build_scene_inventory([season_safe_dir] if products is None else products)
    if not inv:
        print(f"[WARN] No scenes found in {season_safe_dir}; skipping.")
        return

    # Choose reference grid from the first scene that has B02
    ref_b02_path: Optional[Path] = None
    for items in inv.values():
        if "B02" in items:
            ref_b02_path = items["B02"]
            break
    if ref_b02_path is None:
        print(f"[WARN] No B02 JP2 files found in {season_safe_dir}; skipping.")
        return
    ref = open_ref_da(ref_b02_path)

    # Collect per-band stacks (each element is one scene)
    band_stacks: Dict[str, List[xr.DataArray]] = {b: [] for b in ALL_BANDS}
    scl_mask_values_list = list(SCL_MASK_VALUES)  # avoid recreating each time

    # Iterate scenes; for each band present, reproject & mask (reuse SCL per scene)
    for gk, items in inv.items():
        scl_mask = None
        try:
            scl = reproject_to_ref(items["SCL"], ref, Resampling.nearest)
            scl_mask = scl.isnull() | scl.isin(scl_mask_values_list)
        except Exception as exc:
            if scaled:
                print(f"[WARN] Skipping scene without a usable SCL mask: {gk} ({exc})")
                continue
            print(f"[WARN] Using unmasked scene {gk}: {exc}")

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
    for b in ALL_BANDS:
        if b in med_map:
            ordered.append(med_map[b])
        else:
            ordered.append(xr.full_like(ref, np.nan, dtype="float32"))

    out = xr.concat(ordered, dim="band").assign_coords({"band": np.arange(1, len(ALL_BANDS) + 1)})
    out = out.rio.write_crs(ref.rio.crs).rio.write_transform(ref.rio.transform())

    if scaled:
        out = (out / 10000).astype("float32").rio.write_nodata(np.nan)
    else:
        out = (out.fillna(nodata_val).round().clip(min=0, max=65535)
               .astype("uint16").rio.write_nodata(nodata_val))
    out.attrs["long_name"] = tuple(ALL_BANDS)

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    out.rio.to_raster(
        out_tif,
        dtype=str(out.dtype),
        tiled=True,
        compress="deflate",  # change to "zstd" if your GDAL supports it
        predictor=3 if scaled else 2,
        BIGTIFF="IF_SAFER",
    )
    print(f"[OK] Saved {out_tif} ({out.dtype}; bands: {', '.join(ALL_BANDS)}; nodata={out.rio.nodata})")


# -------- multi-tile utils --------
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
    mx.add_argument("--tiles-file", "--tiles_txt", dest="tiles_txt", type=Path,
                    help="Path to a text file with tile IDs (one per line or comma-separated)")

    ap.add_argument("--season", choices=SEASON_ORDER,
                    help="Process a single season (M1..M4). If omitted, processes all.")
    ap.add_argument("--chunk", type=int, default=CHUNK_X,
                    help="Spatial chunk size for x and y (default 1024).")
    ap.add_argument("--nodata", type=int, default=NODATA_U16,
                    help="Nodata value for uint16 output (0 or 65535).")

    args = ap.parse_args()
    if args.chunk < 1:
        ap.error("--chunk must be positive")
    if not 0 <= args.nodata <= 65535:
        ap.error("--nodata must be between 0 and 65535")

    CHUNK_X = CHUNK_Y = int(args.chunk)
    NODATA_U16 = int(args.nodata)

    seasons = [args.season] if args.season else list(SEASON_ORDER)

    # Build tile list
    if args.tile:
        tiles = [normalize_tile(args.tile)]
    else:
        tiles = read_tile_file(args.tiles_txt)
        if not tiles:
            raise SystemExit(f"No tiles found in {args.tiles_txt}")

    root = Path(args.root)
    year = int(args.year)

    for t in tiles:
        process_one_tile(root, year, t, seasons, nodata_val=NODATA_U16)


if __name__ == "__main__":
    main()
