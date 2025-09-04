#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from osgeo import gdal
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
    res = "10m" if band in BANDS_10M else "20m"
    pats = [f"**/*_{band}_{res}.jp2", f"**/*_{band}_{res.upper()}.jp2"]
    out: List[Path] = []
    for p in pats:
        out += list(root.glob(p))
    return sorted(out)


def find_granule_dir(path: Path) -> Optional[Path]:
    cur = path
    for _ in range(12):
        if cur.parent and cur.parent.name == "GRANULE" and cur.name.startswith("L2A_"):
            return cur
        cur = cur.parent
    return None


def find_scl_for_band(band_path: Path) -> Optional[Path]:
    g = find_granule_dir(band_path)
    if g is None:
        return None
    for rel in ("IMG_DATA/R20m/*SCL*20m*.jp2", "QI_DATA/*SCL*20m*.jp2"):
        m = list(g.glob(rel))
        if m:
            m.sort()
            return m[0]
    return None


# -------- IO / reprojection --------
def open_ref_da(b02_path: Path) -> xr.DataArray:
    da = rxr.open_rasterio(b02_path, chunks={"band": 1, "x": CHUNK_X, "y": CHUNK_Y}).squeeze("band", drop=True)
    if da.rio.crs is None:
        raise ValueError(f"No CRS found in {b02_path}")
    da.rio.write_crs(da.rio.crs, inplace=True)
    return da.chunk({"x": CHUNK_X, "y": CHUNK_Y})


def reproject_to_ref(path: Path, ref: xr.DataArray, resampling: Resampling) -> xr.DataArray:
    da = rxr.open_rasterio(path, chunks={"band": 1, "x": CHUNK_X, "y": CHUNK_Y}).squeeze("band", drop=True)
    da = da.rio.reproject_match(ref, resampling=resampling)
    return da.chunk({"x": CHUNK_X, "y": CHUNK_Y})


# -------- processing --------
def build_season_median(season_safe_dir: Path, out_tif: Path, nodata_val: int) -> None:
    # reference grid from first available B02
    b02s = find_band_files(season_safe_dir, "B02")
    if not b02s:
        print(f"[WARN] No B02 JP2 files found in {season_safe_dir}; skipping.")
        return
    ref = open_ref_da(b02s[0])

    # Collect per-band stacks (each element is one scene)
    band_stacks: Dict[str, List[xr.DataArray]] = {b: [] for b in ALL_BANDS}

    for band in ALL_BANDS:
        paths = find_band_files(season_safe_dir, band)
        if not paths:
            print(f"[WARN] No files for {band} in {season_safe_dir}.")
            continue

        resamp = Resampling.bilinear if band in BANDS_20M else Resampling.nearest

        for p in paths:
            try:
                # reproject band to 10 m ref grid; KEEP DN values (no /10000)
                da = reproject_to_ref(p, ref, resamp).astype("float32")

                # per-granule SCL mask
                scl_path = find_scl_for_band(p)
                if scl_path is not None:
                    scl = reproject_to_ref(scl_path, ref, Resampling.nearest)
                    scl_mask = scl.isin(list(SCL_MASK_VALUES))  # True where cloudy/invalid
                    da = da.where(~scl_mask)
                else:
                    print(f"[INFO] No SCL found for {p.name}; using unmasked data for this scene.")

                band_stacks[band].append(da)

            except Exception as e:
                print(f"[SKIP] {p.name}: {e}")

    # Reduce with quantile(0.5) (median) per band, chunk-friendly
    medians: List[xr.DataArray] = []
    names: List[str] = []
    for b, lst in band_stacks.items():
        if not lst:
            continue
        stack = xr.concat(lst, dim="time").chunk({"time": -1, "x": CHUNK_X, "y": CHUNK_Y})
        # Pass q=[0.5] so a 'quantile' dim always exists; then select it
        q = stack.quantile(q=[0.5], dim="time", skipna=True)
        med = q.sel(quantile=0.5)
        medians.append(med)
        names.append(b)

    if not medians:
        print(f"[WARN] No valid data after masking in {season_safe_dir}; skipping.")
        return

    # --- enforce EXACT 7 bands and correct order ---
    med_map = {n: m for n, m in zip(names, medians)}
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
        out.fillna(nodata_val)      # replace NaNs with nodata (e.g., 0)
           .round()                 # round to nearest integer
           .clip(min=0, max=65535)  # clamp to uint16 range
           .astype("uint16")        # cast to uint16
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


def main():
    global CHUNK_X, CHUNK_Y, NODATA_U16  # must be first in main()
    ap = argparse.ArgumentParser(
        description="Process S2 L2A JP2 .SAFE into seasonal cloud-free median mosaics (7 bands @10 m, DN uint16)."
    )
    ap.add_argument("--root", type=Path, required=True, help="Root folder used by downloader (e.g., ./cdse_s2_2018)")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--tile", required=True)
    ap.add_argument("--season", choices=SEASON_ORDER, help="Process a single season (M1..M4). If omitted, processes all.")
    ap.add_argument("--chunk", type=int, default=CHUNK_X, help="Spatial chunk size for x and y (default 1024).")
    ap.add_argument("--nodata", type=int, default=NODATA_U16, help="Nodata value for uint16 output (0 or 65535).")
    args = ap.parse_args()

    CHUNK_X = CHUNK_Y = int(args.chunk)
    NODATA_U16 = int(args.nodata)

    seasons = (args.season,) if args.season else SEASON_ORDER
    for label in seasons:
        season_dir = Path(args.root) / f"{args.year}" / args.tile / label
        safe_dir = season_dir / "SAFE"
        out_tif = season_dir / f"mosaic_{args.year}_{label}_{args.tile}.tif"
        print(f"=== {label} -> {safe_dir} ===")
        build_season_median(safe_dir, out_tif, nodata_val=NODATA_U16)


if __name__ == "__main__":
    main()
