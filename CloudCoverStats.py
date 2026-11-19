#!/usr/bin/env python3
from __future__ import annotations
import argparse
import datetime as dt
import getpass
from pathlib import Path
import pandas as pd

from multi_tile_down_seasons import CDSEAuth, ODATA_BASE, SEASONS, cdse_request, odata_filter, read_tile_file

CLOUD_THRESHOLDS = [20, 30, 40, 50, 60, 70, 80]

def count_products(tile: str, start: dt.datetime, end: dt.datetime, max_cloud: float, auth: CDSEAuth) -> int:
    """
    Server-side count with $count=true & $top=0. Returns an integer from @odata.count.
    """
    params = {
        "$filter": odata_filter(tile, start, end, max_cloud),
        "$count": "true",
        "$top": 0,
        "$select": "Id"
    }
    r = cdse_request("GET", ODATA_BASE, auth, timeout=180, params=params)
    js = r.json()
    return int(js.get("@odata.count", 0))

def build_table_for_window(label: str, start: dt.datetime, end: dt.datetime,
                           tiles: list[str], auth: CDSEAuth) -> pd.DataFrame:
    """
    Build a DataFrame with one row per tile and columns for cumulative counts at
    20,30,...,80% cloud thresholds for the given time window.
    """
    rows = []
    print(f"\n=== Building table for {label}: {start.date()} -> {end.date()} ===")
    for tile in tiles:
        print(f"  Tile {tile}")
        row = {"Tile": tile}
        for thr in CLOUD_THRESHOLDS:
            n = count_products(tile, start, end, thr, auth)
            row[f"<= {thr}%"] = n
            print(f"    <= {thr}% : {n}")
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("Tile").reset_index(drop=True)
    return df

# ----------- CLI -----------
def main():
    ap = argparse.ArgumentParser(
        description="Create per-season (M1..M4) tables of S2 L2A counts by cloud-cover thresholds (CDSE)."
    )
    ap.add_argument("--tiles-file", type=Path, required=True,
                    help="Path to .txt with tile codes separated by whitespace or commas, with optional # comments.")
    ap.add_argument("--year", type=int, required=True,
                    help="Seasonal year to summarize (creates tables for M1..M4).")
    ap.add_argument("--username")
    ap.add_argument("--password")
    ap.add_argument("--out-dir", type=Path, default=Path("s2_counts_output"),
                    help="Directory where CSVs will be saved.")
    args = ap.parse_args()

    tiles = read_tile_file(args.tiles_file)
    if not tiles:
        raise SystemExit(f"No tiles found in {args.tiles_file}")

    user = args.username or input("CDSE username (email): ")
    pw   = args.password  or getpass.getpass("CDSE password: ")
    auth = CDSEAuth(user, pw)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Build one table per season
    for label, start, end in SEASONS(args.year):
        df = build_table_for_window(label, start, end, tiles, auth)
        csv_path = args.out_dir / f"counts_{label}_{args.year}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  -> Saved CSV: {csv_path.resolve()}")

if __name__ == "__main__":
    main()

