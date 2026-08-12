"""
Surface model: object heights above ground, and what those objects are.

Three products, each built from the two rasters acquired in notebook 02.

**nDSM** = DSM - DTM. Height of everything standing on the ground.

**Water** comes from ESA WorldCover, not from the elevation data. An earlier
version thresholded nDSM below zero, since lidar over water is unreliable while
DTM production interpolates across it — but that detects two processing chains
disagreeing, which also happens at bridge undersides and steep quay walls, and
the threshold was arbitrary. See `landcover.py`.

**Classification.** The project's question is not only whether a sightline is
blocked but by what, because trees and buildings behave differently: a 25 m
poplar at the far bank subtends 2.8° and a 80 m tower subtends 9.4°, against a
sun at 7.7°. GRB footprints supply the distinction, since the lidar
classification does not (it carries only ground, water and unclassified).

Rasters are processed in row blocks. The full mosaic is ~72 Mpx; holding
several float32 copies at once is avoidable and there is no reason to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window

NODATA = -9999.0

# Classification codes. uint8, so 0 is reserved for "no data".
CLASS_NODATA = 0
CLASS_GROUND = 1
CLASS_WATER = 2
CLASS_VEGETATION = 3
CLASS_BUILDING = 4
CLASS_UNKNOWN = 5           # tall, but outside GRB footprint coverage

CLASS_LABELS = {
    CLASS_NODATA: "nodata",
    CLASS_GROUND: "ground",
    CLASS_WATER: "water",
    CLASS_VEGETATION: "vegetation",
    CLASS_BUILDING: "building",
    CLASS_UNKNOWN: "tall, unattributed",
}

# These project-specific colours distinguish the six classes in notebook 03's
# diagnostic image. They do not affect classification or horizon calculations.
CLASS_COLOURS = {
    CLASS_NODATA: "#B8C4D8",
    CLASS_GROUND: "#45516E",
    CLASS_WATER: "#56B4E9",
    CLASS_VEGETATION: "#3DBE91",
    CLASS_BUILDING: "#CC79A7",
    CLASS_UNKNOWN: "#E69F00",
}

# An object must exceed this to count as an obstruction rather than ground
# clutter. Kerbs, parked cars and lidar noise sit below it.
VEG_MIN_HEIGHT_M = 3.0


def grid_of(path: Path) -> tuple:
    """(transform, width, height) — the identity of a raster's grid."""
    with rasterio.open(path) as src:
        return (src.transform, src.width, src.height)


def is_stale(product: Path, reference: Path) -> bool:
    """Should `product` be rebuilt because `reference` has changed shape?

    Derived rasters were originally cached on existence alone. That breaks the
    moment an observation point is added: notebook 02 then fetches more tiles,
    the DSM mosaic grows, and every product built from the older, smaller
    mosaic is silently reused at the wrong extent. No error, just an analysis
    quietly covering less ground than it claims.

    Comparing grids instead makes the cache correct rather than merely fast.
    """
    product, reference = Path(product), Path(reference)
    if not product.exists() or product.stat().st_size == 0:
        return True
    try:
        return grid_of(product) != grid_of(reference)
    except rasterio.errors.RasterioIOError:
        return True


def _output_profile(src: rasterio.DatasetReader, **overrides) -> dict:
    """A writable GeoTIFF profile derived from a source that may be a VRT."""
    profile = src.profile.copy()
    # A VRT carries driver="VRT" and block sizes that do not apply to GeoTIFF.
    for key in ("blockxsize", "blockysize", "tiled", "compress", "interleave",
                "photometric"):
        profile.pop(key, None)
    profile.update(driver="GTiff", tiled=True, blockxsize=256, blockysize=256,
                   compress="deflate", predictor=2, BIGTIFF="IF_SAFER")
    profile.update(**overrides)
    return profile


def compute_ndsm(dsm_path: Path, dtm_path: Path, out_path: Path,
                 block_rows: int = 1024) -> Path:
    """Write nDSM = DSM - DTM as a float32 GeoTIFF.

    Nodata in either input propagates to nodata in the output; a height above
    an unknown ground level is not a height.
    """
    dsm_path, dtm_path, out_path = Path(dsm_path), Path(dtm_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dsm_path) as dsm, rasterio.open(dtm_path) as dtm:
        if (dsm.width, dsm.height) != (dtm.width, dtm.height) or \
                dsm.transform != dtm.transform:
            raise ValueError(
                "DSM and DTM are not on the same grid; nDSM would be meaningless")

        profile = _output_profile(dsm, dtype="float32", count=1, nodata=NODATA)
        with rasterio.open(out_path, "w", **profile) as dst:
            for row0 in range(0, dsm.height, block_rows):
                rows = min(block_rows, dsm.height - row0)
                win = Window(0, row0, dsm.width, rows)
                a = dsm.read(1, window=win, masked=True)
                b = dtm.read(1, window=win, masked=True)
                diff = a - b
                dst.write(np.ma.filled(diff, NODATA).astype("float32"),
                          1, window=win)
    return out_path


def rasterize_footprints(geoms: list[dict], like_path: Path,
                         out_path: Path) -> Path:
    """Burn building footprints onto the grid of `like_path` as uint8 0/1.

    `all_touched=True` so a footprint edge crossing a pixel marks that pixel.
    Under-marking would misattribute a building's own wall as vegetation; the
    cost is at most one pixel of over-reach, which is below the accuracy of
    everything else here.
    """
    like_path, out_path = Path(like_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(like_path) as like:
        profile = _output_profile(like, dtype="uint8", count=1, nodata=0)
        shape = (like.height, like.width)
        transform = like.transform

    burned = (rasterize([(g, 1) for g in geoms], out_shape=shape,
                        transform=transform, fill=0, dtype="uint8",
                        all_touched=True)
              if geoms else np.zeros(shape, dtype="uint8"))

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(burned, 1)
    return out_path


def classify(ndsm_path: Path, buildings_path: Path, out_path: Path,
             water_path: Path | None = None,
             coverage_bounds=None,
             veg_min_height: float = VEG_MIN_HEIGHT_M,
             block_rows: int = 1024) -> Path:
    """Classify every pixel from nDSM, a building mask and a water mask.

    Water is supplied as a raster from `landcover.py` rather than inferred from
    the nDSM. It is applied last so it overrides everything else: a pixel ESA
    WorldCover calls permanent water is water, whatever the two elevation models
    happen to disagree about there.

    Footprints are tested before the vegetation threshold, because a footprint
    is authoritative about what a tall object is. Anything tall outside
    `coverage_bounds` becomes CLASS_UNKNOWN rather than being asserted to be
    vegetation — beyond the area where footprints were fetched, the absence of
    a footprint carries no information.
    """
    ndsm_path, buildings_path, out_path = (Path(ndsm_path), Path(buildings_path),
                                           Path(out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    water_src = rasterio.open(water_path) if water_path is not None else None

    with rasterio.open(ndsm_path) as ndsm, rasterio.open(buildings_path) as bld:
        profile = _output_profile(ndsm, dtype="uint8", count=1,
                                  nodata=CLASS_NODATA)
        with rasterio.open(out_path, "w", **profile) as dst:
            for row0 in range(0, ndsm.height, block_rows):
                rows = min(block_rows, ndsm.height - row0)
                win = Window(0, row0, ndsm.width, rows)

                h = ndsm.read(1, window=win, masked=True)
                is_building = bld.read(1, window=win) == 1
                valid = ~np.ma.getmaskarray(h)
                hv = np.ma.filled(h, 0.0)

                out = np.full(hv.shape, CLASS_NODATA, dtype="uint8")
                out[valid] = CLASS_GROUND

                tall = valid & (hv >= veg_min_height)
                out[tall] = CLASS_VEGETATION
                out[tall & is_building] = CLASS_BUILDING

                if coverage_bounds is not None:
                    xs, ys = window_coords(ndsm, win, shape=hv.shape)
                    inside = ((xs >= coverage_bounds.xmin) & (xs <= coverage_bounds.xmax)
                              & (ys >= coverage_bounds.ymin) & (ys <= coverage_bounds.ymax))
                    out[tall & ~is_building & ~inside] = CLASS_UNKNOWN

                # Water last: an authoritative land cover class overrides
                # whatever the elevation models imply.
                if water_src is not None:
                    is_water = water_src.read(1, window=win) == 1
                    out[valid & is_water] = CLASS_WATER

                dst.write(out, 1, window=win)

    if water_src is not None:
        water_src.close()
    return out_path


def window_coords(src: rasterio.DatasetReader, win: Window, shape=None):
    """Pixel-centre coordinates for a window, as broadcastable 1D arrays.

    Returns (xs, ys) shaped (1, W) and (H, 1) so they broadcast against a block
    without materialising two full meshgrids.

    Pass `shape` as the shape of the array actually read. This matters:
    `rasterio.windows.from_bounds` returns a window with *float* offsets and
    sizes, while `read()` rounds it to whole pixels. Deriving the coordinate
    count from `win.width` / `win.height` therefore produces an array one pixel
    larger than the data in each dimension, and every subsequent broadcast
    fails — or worse, silently misaligns.

    `rasterio.transform.xy` is also avoided here: given 2D row/col arrays it
    returns flattened lists, which breaks broadcasting in a different way.
    """
    t = rasterio.windows.transform(win, src.transform)
    if t.b != 0 or t.d != 0:
        raise NotImplementedError("rotated transforms are not supported")

    height, width = shape if shape is not None else (int(win.height), int(win.width))
    xs = (t.c + (np.arange(width) + 0.5) * t.a)[np.newaxis, :]
    ys = (t.f + (np.arange(height) + 0.5) * t.e)[:, np.newaxis]
    return xs, ys


# Kept as a private alias; internal blockwise callers use integer windows.
_block_coords = window_coords


def class_summary(classes_path: Path, block_rows: int = 2048) -> dict:
    """Pixel counts per class, accumulated blockwise."""
    counts: dict[int, int] = {}
    with rasterio.open(classes_path) as src:
        for row0 in range(0, src.height, block_rows):
            rows = min(block_rows, src.height - row0)
            block = src.read(1, window=Window(0, row0, src.width, rows))
            vals, n = np.unique(block, return_counts=True)
            for v, c in zip(vals.tolist(), n.tolist()):
                counts[v] = counts.get(v, 0) + c
    return counts
