"""
Water from ESA WorldCover, rather than inferred from the elevation models.

Earlier versions derived water by thresholding nDSM = DSM - DTM below zero, on
the reasoning that lidar over water is unreliable while DTM production
interpolates across it. That does produce a water-shaped region, but it is a
side effect of two processing chains disagreeing, not a measurement of water.
It picks up bridge undersides, steep quay walls and lidar noise along with the
river, and its threshold is arbitrary.

ESA WorldCover is a purpose-built global land cover product at 10 m, derived
from Sentinel-1 and Sentinel-2, with an explicit permanent-water class. Using it
replaces a heuristic with a dataset.

Access
------
Public COGs on S3, no credentials, 3x3 degree tiles in EPSG:4326:

    s3://esa-worldcover/v200/2021/map/ESA_WorldCover_10m_2021_v200_<TILE>_Map.tif

`<TILE>` is the south-west corner of the tile, floored to a multiple of 3, as
`N51E003`. Because they are COGs, GDAL fetches only the byte ranges covering
our area of interest over HTTP — there is no need to download the ~100 MB tile.

Resolution note: 10 m against our 1 m grid means the water boundary is accurate
to roughly one 10 m cell. For masking a 450 m wide river that is immaterial. It
would matter if we were measuring bank position, which we are not.

Licence: CC-BY 4.0. Attribution: "© ESA WorldCover project 2021 / Contains
modified Copernicus Sentinel data (2021) processed by ESA WorldCover
consortium".
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds

S3_BASE = "https://esa-worldcover.s3.amazonaws.com"
DEFAULT_VERSION = "v200"
DEFAULT_YEAR = 2021
TILE_DEG = 3

# ESA WorldCover class codes (Product User Manual).
CLASSES = {
    10: "tree cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built-up",
    60: "bare / sparse vegetation",
    70: "snow and ice",
    80: "permanent water bodies",
    90: "herbaceous wetland",
    95: "mangroves",
    100: "moss and lichen",
}
CLASS_WATER = 80

# GDAL settings for reading remote COGs efficiently: do not list the bucket
# directory, and retry transient HTTP failures.
GDAL_HTTP_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
}


def tile_id(lat: float, lon: float) -> str:
    """WorldCover tile name for a coordinate, e.g. (51.2, 4.4) -> 'N51E003'.

    Tiles are named for their south-west corner, floored to a multiple of 3.
    Latitude takes two digits, longitude three.
    """
    tlat = int(math.floor(lat / TILE_DEG) * TILE_DEG)
    tlon = int(math.floor(lon / TILE_DEG) * TILE_DEG)
    ns = "N" if tlat >= 0 else "S"
    ew = "E" if tlon >= 0 else "W"
    return f"{ns}{abs(tlat):02d}{ew}{abs(tlon):03d}"


def tiles_for_bbox(west: float, south: float, east: float, north: float) -> list[str]:
    """Every WorldCover tile intersecting a WGS84 bounding box."""
    lat0 = int(math.floor(south / TILE_DEG) * TILE_DEG)
    lon0 = int(math.floor(west / TILE_DEG) * TILE_DEG)
    out = []
    lat = lat0
    while lat <= north:
        lon = lon0
        while lon <= east:
            out.append(tile_id(lat, lon))
            lon += TILE_DEG
        lat += TILE_DEG
    return sorted(set(out))


def worldcover_url(tile: str, version: str = DEFAULT_VERSION,
                   year: int = DEFAULT_YEAR) -> str:
    """Public HTTPS URL of one WorldCover map tile."""
    return (f"{S3_BASE}/{version}/{year}/map/"
            f"ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif")


def fetch_landcover(like_path: Path, out_path: Path,
                    version: str = DEFAULT_VERSION, year: int = DEFAULT_YEAR,
                    buffer_deg: float = 0.05,
                    overwrite: bool = False,
                    progress: bool = True) -> Path:
    """Reproject WorldCover onto the grid of `like_path`.

    Nearest-neighbour resampling, because the values are class codes: averaging
    class 50 and class 80 would produce class 65, which means nothing.

    Only the byte ranges covering the area of interest are fetched, so this
    transfers a few MB rather than the full tile.
    """
    like_path, out_path = Path(like_path), Path(out_path)
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        if progress:
            print(f"  {out_path.name}: from cache")
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(like_path) as like:
        dst_crs, dst_transform = like.crs, like.transform
        dst_shape = (like.height, like.width)
        bounds = like.bounds
        profile = like.profile.copy()

    for key in ("blockxsize", "blockysize", "tiled", "compress", "interleave",
                "photometric", "predictor"):
        profile.pop(key, None)
    profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0,
                   tiled=True, blockxsize=256, blockysize=256,
                   compress="deflate", BIGTIFF="IF_SAFER")

    west, south, east, north = transform_bounds(
        dst_crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top)
    west, south = west - buffer_deg, south - buffer_deg
    east, north = east + buffer_deg, north + buffer_deg

    dst = np.zeros(dst_shape, dtype="uint8")
    tiles = tiles_for_bbox(west, south, east, north)
    if progress:
        print(f"  WorldCover tiles needed: {tiles}")

    with rasterio.Env(**GDAL_HTTP_ENV):
        for tile in tiles:
            url = worldcover_url(tile, version, year)
            try:
                src = rasterio.open(url)
            except Exception as exc:
                raise RuntimeError(
                    f"could not open {url}\n"
                    f"  Tile names encode the south-west corner floored to 3 "
                    f"degrees (e.g. N51E003). If the naming or the version path "
                    f"has changed, correct worldcover_url() in landcover.py.\n"
                    f"  underlying error: {exc}") from None

            with src:
                win = window_from_bounds(west, south, east, north,
                                         transform=src.transform)
                win = win.round_offsets().round_lengths()
                win = win.intersection(Window(0, 0, src.width, src.height))
                if win.width <= 0 or win.height <= 0:
                    continue
                arr = src.read(1, window=win)
                src_transform = src.window_transform(win)
                src_crs = src.crs
                if progress:
                    print(f"  {tile}: read {arr.shape[1]}x{arr.shape[0]} px")

            patch = np.zeros(dst_shape, dtype="uint8")
            reproject(source=arr, destination=patch,
                      src_transform=src_transform, src_crs=src_crs,
                      dst_transform=dst_transform, dst_crs=dst_crs,
                      resampling=Resampling.nearest,
                      src_nodata=0, dst_nodata=0)
            # Tiles do not overlap; maximum is a simple, order-free merge.
            dst = np.maximum(dst, patch)

    with rasterio.open(out_path, "w", **profile) as out:
        out.write(dst, 1)
    return out_path


def water_mask(landcover_path: Path, out_path: Path,
               water_class: int = CLASS_WATER) -> Path:
    """Extract a 0/1 water mask from a WorldCover raster."""
    landcover_path, out_path = Path(landcover_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(landcover_path) as src:
        profile = src.profile.copy()
        profile.update(dtype="uint8", count=1, nodata=0)
        arr = (src.read(1) == water_class).astype("uint8")

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)
    return out_path


def class_counts(landcover_path: Path) -> dict:
    """Pixel counts per WorldCover class."""
    with rasterio.open(landcover_path) as src:
        vals, n = np.unique(src.read(1), return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, n)}
