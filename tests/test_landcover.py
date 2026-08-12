"""Tests for ESA WorldCover access and the water mask.

Tile-name construction and URL building are pure functions and tested directly.
The remote COG read is not exercised here — it needs network — but it is
covered by the notebook harness, which stubs the fetch.

Run with:  pytest -v
"""


import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


from eclipse_viewshed import landcover


# ------------------------------------------------------------- tile naming

def test_antwerp_falls_in_N51E003():
    """Tiles are named for the south-west corner floored to 3 degrees.
    Antwerp at 51.2 N, 4.4 E therefore sits in the tile starting 51 N, 3 E."""
    assert landcover.tile_id(51.21193309444983, 4.388558346421922) == "N51E003"


def test_tile_id_floors_to_multiples_of_three():
    assert landcover.tile_id(51.0, 3.0) == "N51E003"
    assert landcover.tile_id(53.9, 5.9) == "N51E003"
    assert landcover.tile_id(54.0, 6.0) == "N54E006"


def test_tile_id_pads_digits():
    """Latitude two digits, longitude three."""
    assert landcover.tile_id(3.5, 4.5) == "N03E003"
    assert landcover.tile_id(0.0, 0.0) == "N00E000"


def test_tile_id_southern_and_western_hemispheres():
    assert landcover.tile_id(-1.0, -1.0) == "S03W003"
    assert landcover.tile_id(-33.9, -70.6) == "S36W072"


def test_tiles_for_bbox_single_tile():
    """The project AOI sits well inside one tile."""
    assert landcover.tiles_for_bbox(4.24, 51.21, 4.41, 51.26) == ["N51E003"]


def test_tiles_for_bbox_spanning_a_boundary():
    tiles = landcover.tiles_for_bbox(2.5, 50.5, 3.5, 51.5)
    assert "N48E000" in tiles or "N48E003" in tiles
    assert len(tiles) >= 2
    assert len(set(tiles)) == len(tiles)


# ---------------------------------------------------------------- URL

def test_worldcover_url_shape():
    url = landcover.worldcover_url("N51E003")
    assert url.startswith("https://esa-worldcover.s3.amazonaws.com/")
    assert "/v200/2021/map/" in url
    assert url.endswith("ESA_WorldCover_10m_2021_v200_N51E003_Map.tif")


def test_worldcover_url_honours_version_and_year():
    url = landcover.worldcover_url("N51E003", version="v100", year=2020)
    assert "/v100/2020/map/" in url
    assert url.endswith("ESA_WorldCover_10m_2020_v100_N51E003_Map.tif")


# ---------------------------------------------------------- water mask

@pytest.fixture
def landcover_raster(tmp_path):
    """A small WorldCover-like raster: built-up, tree cover, a water channel."""
    arr = np.full((40, 40), 50, dtype="uint8")      # built-up
    arr[30:, :] = 10                                # tree cover
    # Water written last so the channel is an unbroken 40 x 10 band; writing it
    # before the tree rows would clip it and make the count assertion wrong.
    arr[:, 10:20] = landcover.CLASS_WATER           # permanent water
    p = tmp_path / "lc.tif"
    prof = dict(driver="GTiff", height=40, width=40, count=1, dtype="uint8",
                crs="EPSG:31370", nodata=0,
                transform=from_origin(151000, 211040, 1, 1))
    with rasterio.open(p, "w", **prof) as dst:
        dst.write(arr, 1)
    return p


def test_water_mask_selects_only_class_80(landcover_raster, tmp_path):
    out = landcover.water_mask(landcover_raster, tmp_path / "w.tif")
    with rasterio.open(out) as src:
        m = src.read(1)
    assert m[5, 15] == 1
    assert m[5, 5] == 0
    assert m[35, 5] == 0
    assert set(np.unique(m)) <= {0, 1}


def test_water_mask_preserves_grid(landcover_raster, tmp_path):
    out = landcover.water_mask(landcover_raster, tmp_path / "w.tif")
    with rasterio.open(landcover_raster) as a, rasterio.open(out) as b:
        assert a.transform == b.transform
        assert (a.width, a.height) == (b.width, b.height)


def test_class_counts_totals_all_pixels(landcover_raster):
    counts = landcover.class_counts(landcover_raster)
    assert sum(counts.values()) == 40 * 40
    assert counts[landcover.CLASS_WATER] == 40 * 10


def test_water_class_code_is_documented():
    assert landcover.CLASS_WATER == 80
    assert "water" in landcover.CLASSES[landcover.CLASS_WATER]


# ---------------------------------------------------------- cache naming

def test_cache_name_encodes_the_bounds():
    """A fixed filename would silently reuse footprints from a previous area
    of interest after the primary observer moves."""
    from eclipse_viewshed import vector
    from eclipse_viewshed.aoi import Bounds

    a = vector.cache_name(vector.LAYER_BUILDINGS, Bounds(1, 2, 3, 4))
    b = vector.cache_name(vector.LAYER_BUILDINGS, Bounds(1, 2, 3, 5))
    assert a != b
    assert a.endswith(".geojson")
    assert ":" not in a          # must be a legal filename on Windows
