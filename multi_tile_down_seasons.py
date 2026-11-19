#!/usr/bin/env python3
from __future__ import annotations
import argparse
import datetime as dt
import getpass
from pathlib import Path
import zipfile
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from tempfile import TemporaryDirectory

import requests
import pandas as pd
from tqdm import tqdm

CDSE_OAUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
ODATA_BASE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
ZIPPER_BASE = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"

SEASONS = lambda y: [
    ("M1", dt.datetime(y-1,12,1), dt.datetime(y,3,1)),
    ("M2", dt.datetime(y,3,1),  dt.datetime(y,6,1)),
    ("M3", dt.datetime(y,6,1),  dt.datetime(y,9,1)),
    ("M4", dt.datetime(y,9,1),  dt.datetime(y,12,1)),
]

def token(username:str, password:str)->str:
    r = requests.post(CDSE_OAUTH_URL, data={
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }, timeout=60)
    r.raise_for_status()
    return r.json()["access_token"]

class CDSEAuth:
    """Thread-safe token holder with auto-refresh."""
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.access_token = None
        self._lock = threading.Lock()
        self.refresh()
    def refresh(self):
        with self._lock:
            self.access_token = token(self.username, self.password)
    def headers(self):
        with self._lock:
            return {"Authorization": f"Bearer {self.access_token}"}

def cdse_request(method: str, url: str, auth: CDSEAuth, *,
                 stream: bool = False, timeout: int = 300,
                 max_retries: int = 4, **kwargs):
    """requests wrapper: auto token refresh on 401/403 + backoff retries"""
    last_exc = None
    for attempt in range(max_retries):
        try:
            r = requests.request(method, url, headers=auth.headers(),
                                 stream=stream, timeout=timeout, **kwargs)
            if r.status_code in (401, 403):
                auth.refresh()
                r = requests.request(method, url, headers=auth.headers(),
                                     stream=stream, timeout=timeout, **kwargs)
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            last_exc = e
            code = getattr(e.response, "status_code", None)
            if code in (401, 403):
                if attempt == max_retries - 1:
                    raise
            elif code is not None and 400 <= code < 500 and code != 429:
                raise
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
        sleep_s = min(60, (2 ** attempt)) + random.uniform(0, 0.5)
        print(f"Request retry {attempt+1}/{max_retries}. Backing off {sleep_s:.1f}s")
        time.sleep(sleep_s)
    raise last_exc or RuntimeError(f"Failed after {max_retries} attempts: {url}")

def normalize_tile(t: str) -> str:
    t = t.strip().upper()
    return t if t.startswith("T") else ("T" + t)


def read_tile_file(path: Path) -> list[str]:
    """Read tile codes separated by commas or whitespace, with optional comments."""
    tiles = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        tiles.extend(line.partition("#")[0].replace(",", " ").split())
    return list(dict.fromkeys(normalize_tile(tile) for tile in tiles))


def parse_tiles(single: str | None, many: list[str] | None, fpath: Path | None) -> list[str]:
    tiles: set[str] = set()
    if single:
        tiles.add(single)
    if many:
        tiles.update(many)
    if fpath:
        tiles.update(read_tile_file(fpath))
    if not tiles:
        raise SystemExit("You must provide --tile OR --tiles ... OR --tiles-file.")
    return sorted({normalize_tile(t) for t in tiles if t.strip()})

def odata_filter(tile: str, start: dt.datetime, end: dt.datetime, max_cloud: float) -> str:
    return (
        "Collection/Name eq 'SENTINEL-2' "
        "and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') "
        f"and ContentDate/Start ge {start.strftime('%Y-%m-%d')}T00:00:00.000Z "
        f"and ContentDate/Start lt {end.strftime('%Y-%m-%d')}T00:00:00.000Z "
        f"and contains(Name,'_{tile}_') "
        f"and Attributes/OData.CSC.DoubleAttribute/any(att: att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {max_cloud})"
    )


def odata_url(tile: str, start: dt.datetime, end: dt.datetime, max_cloud: float) -> str:
    params = {"$filter": odata_filter(tile, start, end, max_cloud),
              "$orderby": "ContentDate/Start asc", "$top": 200}
    def enc(v:str)->str: return requests.utils.requote_uri(v)
    q = "&".join([f"{k}={enc(str(v))}" for k,v in params.items()])
    return f"{ODATA_BASE}?{q}"

def search(tile, start, end, max_cloud, auth: CDSEAuth) -> pd.DataFrame:
    url = odata_url(tile, start, end, max_cloud)
    out = []
    while url:
        r = cdse_request("GET", url, auth, timeout=180)
        js = r.json()
        out.extend(js.get("value", []))
        url = js.get("@odata.nextLink")
    return pd.DataFrame(out)

def download(pid: str, name: str, dest_zip: Path, auth: CDSEAuth, progress: bool = True):
    if zipfile.is_zipfile(dest_zip):
        return "exists"
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    url = f"{ZIPPER_BASE}({pid})/$value"
    with TemporaryDirectory(dir=dest_zip.parent) as tmp:
        partial = Path(tmp) / dest_zip.name
        with cdse_request("GET", url, auth, stream=True, timeout=600) as r, partial.open("wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            with tqdm(total=total or None, unit="B", unit_scale=True, desc=name, disable=not progress) as p:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
                    p.update(len(chunk))
        if not zipfile.is_zipfile(partial):
            raise IOError("The downloaded response is not a complete ZIP archive.")
        partial.replace(dest_zip)
    return "downloaded"

def extract(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        roots = {name.split("/", 1)[0] for name in z.namelist() if "/" in name}
        safe_roots = {root for root in roots if root.endswith(".SAFE")}
        if len(safe_roots) != 1:
            raise ValueError(f"Expected one top-level .SAFE product in {zip_path}")
        root = safe_roots.pop()
        safe_path = out_dir / root
        if safe_path.exists():
            return safe_path
        with TemporaryDirectory(dir=out_dir) as tmp:
            z.extractall(tmp)
            (Path(tmp) / root).rename(safe_path)
        return safe_path

def main():
    ap = argparse.ArgumentParser(description="Download Sentinel-2 L2A JP2 products by season (CDSE) for one or more tiles")
    # tiles
    ap.add_argument("--tile", help="Single tile (e.g., T34SGH)")
    ap.add_argument("--tiles", nargs="+", help="Multiple tiles (space-separated), e.g., T34SGH T34SGJ")
    ap.add_argument("--tiles-file", type=Path, help="Text file with tiles (one per line or comma-separated)")
    # other params
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("./cdse_s2_output"))
    ap.add_argument("--max-cloud", type=float, default=80.0)
    ap.add_argument("--username")
    ap.add_argument("--password")
    ap.add_argument("--workers", type=int, default=1, help="Parallel downloads per season (use 1–3 to be gentle)")
    args = ap.parse_args()

    tiles = parse_tiles(args.tile, args.tiles, args.tiles_file)
    download_seasons(tiles, args.year, args.outdir, args.max_cloud,
                     args.username, args.password, args.workers)


def download_seasons(tiles, year, outdir, max_cloud, username=None, password=None,
                     workers=1) -> dict[tuple[str, str], list[Path]]:
    user = username or input("CDSE username (email): ")
    pw = password or getpass.getpass("CDSE password: ")
    auth = CDSEAuth(user, pw)

    print(f"Tiles: {', '.join(tiles)} | Year: {year} | Max cloud: {max_cloud}% | Workers: {workers}")

    products = {}
    for tile in tiles:
        print(f"\n========== Tile {tile} ==========")
        for label, start, end in SEASONS(year):
            print(f"=== {label}: {start.date()} -> {end.date()} ===")
            df = search(tile, start, end, max_cloud, auth)
            print(f"Found {len(df)} products.")
            if df.empty:
                continue

            season_dir = outdir / str(year) / tile / label
            raw_dir = season_dir / "raw"
            safe_dir = season_dir / "SAFE"

            def job(row):
                pid, name = row["Id"], row["Name"]
                archive = raw_dir / f"{name}.zip"
                try:
                    download(pid, name, archive, auth, progress=workers <= 1)
                    return extract(archive, safe_dir)
                except Exception as exc:
                    print(f"Warning: failed {name}: {exc}")
                    return None

            rows = df.drop_duplicates(subset="Name").to_dict("records")
            if workers <= 1:
                paths = list(map(job, rows))
            else:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    paths = list(ex.map(job, rows))
            products[tile, label] = [path for path in paths if path is not None]
    return products

if __name__ == "__main__":
    main()
