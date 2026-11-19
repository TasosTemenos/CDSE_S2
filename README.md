# Sentinel-2 seasonal mosaics

Download Sentinel-2 L2A products from the Copernicus Data Space Ecosystem (CDSE), build seasonal median GeoTIFFs per MGRS tile, or count available products by cloud cover.

## Setup

From the project directory, create and activate the Conda environment:

```sh
conda env create -f environment.yml
conda activate cdse-s2
```

The downloader and cloud-count scripts prompt for your CDSE email and password. Processing existing SAFE files works offline. Keep the output path short on Windows and allow disk space for both ZIP archives and extracted products.

## Download and process

Run both commands with the same year and tile list:

```sh
python multi_tile_down_seasons.py --tiles-file tiles.txt --year 2024 --outdir ./cdse_s2_output --max-cloud 80
python s2_median_mosaic_2.py --tiles-file tiles.txt --year 2024 --root ./cdse_s2_output
```

- Use `--tile T34SGH` in either command for a single tile. The downloader also accepts `--tiles T34SGH T34SGJ` and `--workers 2` for parallel downloads within a season. Workers default to 1.
- Tile files accept comma- or whitespace-separated codes and `#` comments. Codes are uppercased and prefixed with `T` when needed. The mosaic script also accepts its original `--tiles_txt` flag.
- Add `--season M2` to process one season, `--chunk 512` to reduce the spatial chunk size from 1024, or `--nodata 65535` to change the uint16 nodata value from 0. Reprojection can still use substantial memory.
- `--max-cloud` is the product-level catalogue filter. The mosaic applies pixel-level Scene Classification (SCL) masking. The standalone processor warns and uses unmasked data if a scene's SCL is missing or unreadable.

For one tile in a single command, use the combined workflow. It processes the products selected by that run and writes float32 values scaled as DN / 10,000 with NaN nodata. It skips scenes without a usable SCL mask. This scaling does not apply product-specific radiometric offsets.

```sh
python cdse_s2_seasonal.py --tile T34SGH --year 2024 --outdir ./cdse_float_output
```

Use separate output roots if you need both formats. Both commands write the same mosaic filename within a given root.

## Count products by cloud cover

```sh
python CloudCoverStats.py --tiles-file tiles.txt --year 2024
```

This queries the catalogue without downloading imagery. It writes `s2_counts_output/counts_M1_2024.csv` through `counts_M4_2024.csv`, with one row per tile and cumulative counts for thresholds from 20% through 80% in 10-point steps. Use `--out-dir` to change the report directory.

## Seasons and outputs

Dates are UTC. Each start is inclusive and each end is exclusive.

| Season | Start | End |
| --- | --- | --- |
| M1 | December 1 of the previous year | March 1 |
| M2 | March 1 | June 1 |
| M3 | June 1 | September 1 |
| M4 | September 1 | December 1 |

For each tile and season, the root contains:

```text
2024/T34SGH/M2/
  raw/                          # Downloaded ZIP archives
  SAFE/                         # Extracted .SAFE product directories
  mosaic_2024_M2_T34SGH.tif
```

The mosaics use the first available B02 grid at 10 m. The seven bands are B02, B03, B04, B08, B8A, B11, and B12, in that order. The processor uses bilinear resampling for 20 m spectral bands and nearest neighbour for SCL. SCL classes 0, 1, 3, 8, 9, 10, and 11 are excluded from the median. Missing bands and pixels without a usable observation are written as nodata.

## Migrating older commands

| Previous script | Replacement |
| --- | --- |
| `cdse_download_seasons.py` | `multi_tile_down_seasons.py` with the same `--tile` arguments |
| `s2_process_seasons.py` | `s2_median_mosaic_2.py` with the same `--root`, year, tile, season, chunk, and nodata arguments |

Run any of the four remaining scripts with `--help` for all options.
