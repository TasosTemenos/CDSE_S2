#!/usr/bin/env python3
"""Download and build seasonal float32 mosaics for a single Sentinel-2 tile."""
from __future__ import annotations

import argparse
from pathlib import Path

from multi_tile_down_seasons import download_seasons, normalize_tile
from s2_median_mosaic_2 import NODATA_U16, build_season_median


def run(tile: str, year: int, outdir: Path, username: str | None,
        password: str | None, max_cloud: float) -> None:
    tile = normalize_tile(tile)
    products = download_seasons([tile], year, outdir, max_cloud, username, password)
    for (_, season), paths in products.items():
        if paths:
            season_dir = outdir / str(year) / tile / season
            output = season_dir / f"mosaic_{year}_{season}_{tile}.tif"
            build_season_median(season_dir / "SAFE", output, NODATA_U16,
                                scaled=True, products=paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile", required=True, help="MGRS tile, e.g. T34SGH")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("./cdse_s2_output"))
    parser.add_argument("--username")
    parser.add_argument("--password")
    parser.add_argument("--max-cloud", type=float, default=80.0)
    args = parser.parse_args()
    run(args.tile, args.year, args.outdir, args.username, args.password, args.max_cloud)


if __name__ == "__main__":
    main()
