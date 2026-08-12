"""Tests for the surface model: nDSM, water detection, classification.

Rasters are synthesised on disk so rasterio does real I/O — the same reason
the notebook harness exists. Classification is checked at the boundaries, since
that is where threshold logic goes wrong.

Run with:  pytest -v
"""


import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


from eclipse_viewshed import surface
from eclipse_viewshed.aoi import Bounds

X0, Y0, N = 151000, 211000, 60


def write_raster(path, arr, nodata=surface.NODATA, dtype="float32"):
    prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                dtype=dtype, crs="EPSG:31370", nodata=nodata,
                transform=from_origin(X0, Y0 + arr.shape[0], 1, 1))
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)
    return path


WATER_COLS = slice(45, 55)


@pytest.fixture
def surfaces(tmp_path):
    """A small scene: flat ground, a building, a tree, a water channel."""
    dtm = np.full((N, N), 5.0, dtype="float32")
    dsm = dtm.copy()
    dsm[10:20, 10:20] += 25.0                  # building, 25 m
    dsm[30:36, 30:36] += 12.0                  # tree, 12 m
    dsm[:, WATER_COLS] = dtm[:, WATER_COLS] - 4  # lidar over water
    return (write_raster(tmp_path / "dsm.tif", dsm),
            write_raster(tmp_path / "dtm.tif", dtm),
            tmp_path)


@pytest.fixture
def water_raster(tmp_path):
    """A water mask as landcover.py would produce it, independent of the nDSM."""
    w = np.zeros((N, N), dtype="uint8")
    w[:, WATER_COLS] = 1
    return write_raster(tmp_path / "water.tif", w, nodata=0, dtype="uint8")


# ------------------------------------------------------------------- nDSM

def test_ndsm_is_dsm_minus_dtm(surfaces):
    dsm_p, dtm_p, tmp = surfaces
    out = surface.compute_ndsm(dsm_p, dtm_p, tmp / "ndsm.tif")
    with rasterio.open(out) as src:
        n = src.read(1, masked=True)
    assert n[15, 15] == pytest.approx(25.0, abs=0.01)
    assert n[33, 33] == pytest.approx(12.0, abs=0.01)
    assert n[5, 5] == pytest.approx(0.0, abs=0.01)
    assert n[5, 50] == pytest.approx(-4.0, abs=0.01)


def test_ndsm_preserves_grid(surfaces):
    dsm_p, dtm_p, tmp = surfaces
    out = surface.compute_ndsm(dsm_p, dtm_p, tmp / "ndsm.tif")
    with rasterio.open(dsm_p) as a, rasterio.open(out) as b:
        assert a.transform == b.transform
        assert (a.width, a.height) == (b.width, b.height)
        assert a.crs == b.crs


def test_ndsm_output_is_geotiff_not_vrt(surfaces):
    """The inputs are VRTs in practice; the profile must be overridden."""
    dsm_p, dtm_p, tmp = surfaces
    out = surface.compute_ndsm(dsm_p, dtm_p, tmp / "ndsm.tif")
    with rasterio.open(out) as src:
        assert src.driver == "GTiff"


def test_ndsm_rejects_mismatched_grids(tmp_path):
    a = write_raster(tmp_path / "a.tif", np.zeros((N, N), "float32"))
    b = write_raster(tmp_path / "b.tif", np.zeros((N // 2, N), "float32"))
    with pytest.raises(ValueError, match="same grid"):
        surface.compute_ndsm(a, b, tmp_path / "out.tif")


def test_ndsm_propagates_nodata(tmp_path):
    dtm = np.full((N, N), 5.0, dtype="float32")
    dsm = dtm.copy()
    dsm[0, 0] = surface.NODATA
    a = write_raster(tmp_path / "dsm.tif", dsm)
    b = write_raster(tmp_path / "dtm.tif", dtm)
    out = surface.compute_ndsm(a, b, tmp_path / "ndsm.tif")
    with rasterio.open(out) as src:
        n = src.read(1, masked=True)
    assert np.ma.getmaskarray(n)[0, 0]


def test_ndsm_blockwise_matches_single_block(surfaces):
    """Block size must not change the result."""
    dsm_p, dtm_p, tmp = surfaces
    big = surface.compute_ndsm(dsm_p, dtm_p, tmp / "a.tif", block_rows=10_000)
    small = surface.compute_ndsm(dsm_p, dtm_p, tmp / "b.tif", block_rows=7)
    with rasterio.open(big) as x, rasterio.open(small) as y:
        assert np.allclose(x.read(1), y.read(1))


# --------------------------------------------------------------- rasterise

def test_rasterize_marks_footprint_interior(surfaces):
    dsm_p, _, tmp = surfaces
    poly = {"type": "Polygon", "coordinates": [[
        [X0 + 10, Y0 + N - 20], [X0 + 20, Y0 + N - 20],
        [X0 + 20, Y0 + N - 10], [X0 + 10, Y0 + N - 10],
        [X0 + 10, Y0 + N - 20]]]}
    out = surface.rasterize_footprints([poly], dsm_p, tmp / "b.tif")
    with rasterio.open(out) as src:
        m = src.read(1)
    assert m[15, 15] == 1
    assert m[50, 5] == 0


def test_rasterize_empty_list_is_all_zero(surfaces):
    dsm_p, _, tmp = surfaces
    out = surface.rasterize_footprints([], dsm_p, tmp / "b.tif")
    with rasterio.open(out) as src:
        assert src.read(1).max() == 0


# ------------------------------------------------------------ classification

@pytest.fixture
def classified(surfaces, water_raster):
    dsm_p, dtm_p, tmp = surfaces
    ndsm = surface.compute_ndsm(dsm_p, dtm_p, tmp / "ndsm.tif")
    poly = {"type": "Polygon", "coordinates": [[
        [X0 + 10, Y0 + N - 20], [X0 + 20, Y0 + N - 20],
        [X0 + 20, Y0 + N - 10], [X0 + 10, Y0 + N - 10],
        [X0 + 10, Y0 + N - 20]]]}
    bmask = surface.rasterize_footprints([poly], ndsm, tmp / "b.tif")
    out = surface.classify(ndsm, bmask, tmp / "classes.tif", water_path=water_raster)
    with rasterio.open(out) as src:
        return src.read(1), out


def test_building_pixels_are_buildings(classified):
    arr, _ = classified
    assert arr[15, 15] == surface.CLASS_BUILDING


def test_tree_pixels_are_vegetation(classified):
    arr, _ = classified
    assert arr[33, 33] == surface.CLASS_VEGETATION


def test_water_pixels_are_water(classified):
    """Water comes from the supplied mask, not from a nDSM threshold."""
    arr, _ = classified
    assert arr[5, 50] == surface.CLASS_WATER


def test_no_water_class_without_a_water_mask(surfaces):
    """Without a mask, nothing is called water however negative the nDSM is.

    Regression guard for the removed heuristic: a negative nDSM alone must no
    longer produce a water classification.
    """
    dsm_p, dtm_p, tmp = surfaces
    ndsm = surface.compute_ndsm(dsm_p, dtm_p, tmp / "ndsm.tif")
    bmask = surface.rasterize_footprints([], ndsm, tmp / "b.tif")
    out = surface.classify(ndsm, bmask, tmp / "c.tif", water_path=None)
    with rasterio.open(out) as src:
        arr = src.read(1)
    assert surface.CLASS_WATER not in np.unique(arr)
    assert arr[5, 50] == surface.CLASS_GROUND


def test_water_overrides_a_tall_object(surfaces, water_raster):
    """The mask is authoritative: if WorldCover says water, water it is."""
    dsm_p, dtm_p, tmp = surfaces
    ndsm = surface.compute_ndsm(dsm_p, dtm_p, tmp / "ndsm.tif")
    bmask = surface.rasterize_footprints([], ndsm, tmp / "b.tif")

    # Force a tall object inside the water columns.
    with rasterio.open(ndsm, "r+") as src:
        a = src.read(1)
        a[20, 50] = 30.0
        src.write(a, 1)

    out = surface.classify(ndsm, bmask, tmp / "c.tif", water_path=water_raster)
    with rasterio.open(out) as src:
        assert src.read(1)[20, 50] == surface.CLASS_WATER


def test_flat_pixels_are_ground(classified):
    arr, _ = classified
    assert arr[5, 5] == surface.CLASS_GROUND


def test_vegetation_threshold_boundary(tmp_path):
    """A 2.9 m object is clutter, a 3.1 m object is an obstruction."""
    dtm = np.full((20, 20), 5.0, dtype="float32")
    dsm = dtm.copy()
    dsm[5, 5] += 2.9
    dsm[10, 10] += 3.1
    a = write_raster(tmp_path / "dsm.tif", dsm)
    b = write_raster(tmp_path / "dtm.tif", dtm)
    ndsm = surface.compute_ndsm(a, b, tmp_path / "ndsm.tif")
    bmask = surface.rasterize_footprints([], ndsm, tmp_path / "b.tif")
    out = surface.classify(ndsm, bmask, tmp_path / "c.tif")
    with rasterio.open(out) as src:
        arr = src.read(1)
    assert arr[5, 5] == surface.CLASS_GROUND
    assert arr[10, 10] == surface.CLASS_VEGETATION


def test_tall_outside_coverage_is_unknown_not_vegetation(surfaces):
    """Beyond the area where footprints were fetched, a missing footprint says
    nothing, so a tall object must not be asserted to be a tree."""
    dsm_p, dtm_p, tmp = surfaces
    ndsm = surface.compute_ndsm(dsm_p, dtm_p, tmp / "ndsm.tif")
    bmask = surface.rasterize_footprints([], ndsm, tmp / "b.tif")

    # Row 0 is the NORTH edge, at y = Y0 + N, and y decreases with row index:
    #     y(row) = Y0 + N - row - 0.5
    # so the building at rows 10-20 sits at y ~211040-211050, and the tree at
    # rows 30-36 sits at y ~211024-211030.
    # This box therefore contains the tree and excludes the building.
    coverage = Bounds(X0, Y0, X0 + N, Y0 + 35)
    out = surface.classify(ndsm, bmask, tmp / "c.tif", coverage_bounds=coverage)
    with rasterio.open(out) as src:
        arr = src.read(1)
    assert arr[15, 15] == surface.CLASS_UNKNOWN
    assert arr[33, 33] == surface.CLASS_VEGETATION


def test_classification_blockwise_is_stable(surfaces):
    dsm_p, dtm_p, tmp = surfaces
    ndsm = surface.compute_ndsm(dsm_p, dtm_p, tmp / "ndsm.tif")
    bmask = surface.rasterize_footprints([], ndsm, tmp / "b.tif")
    a = surface.classify(ndsm, bmask, tmp / "a.tif", block_rows=10_000)
    b = surface.classify(ndsm, bmask, tmp / "b2.tif", block_rows=7)
    with rasterio.open(a) as x, rasterio.open(b) as y:
        assert np.array_equal(x.read(1), y.read(1))


def test_class_summary_counts_all_pixels(classified):
    _, path = classified
    counts = surface.class_summary(path)
    assert sum(counts.values()) == N * N
    assert counts[surface.CLASS_WATER] > 0
    assert counts[surface.CLASS_BUILDING] > 0


def test_every_class_code_has_a_label():
    for code in (surface.CLASS_NODATA, surface.CLASS_GROUND, surface.CLASS_WATER,
                 surface.CLASS_VEGETATION, surface.CLASS_BUILDING,
                 surface.CLASS_UNKNOWN):
        assert code in surface.CLASS_LABELS
        assert code in surface.CLASS_COLOURS


# ---------------------------------------------- grid-aware cache staleness

def test_is_stale_when_missing(tmp_path, surfaces):
    dsm_p, _, _ = surfaces
    assert surface.is_stale(tmp_path / "nope.tif", dsm_p)


def test_is_stale_when_empty(tmp_path, surfaces):
    dsm_p, _, _ = surfaces
    empty = tmp_path / "empty.tif"
    empty.write_bytes(b"")
    assert surface.is_stale(empty, dsm_p)


def test_not_stale_when_grids_match(surfaces, tmp_path):
    dsm_p, dtm_p, tmp = surfaces
    ndsm = surface.compute_ndsm(dsm_p, dtm_p, tmp / "ndsm.tif")
    assert not surface.is_stale(ndsm, dsm_p)


def test_is_stale_when_the_reference_grows(surfaces, tmp_path):
    """The real scenario: adding an observation point grows the DSM mosaic, so
    every product built from the old one must be rebuilt."""
    dsm_p, dtm_p, tmp = surfaces
    ndsm = surface.compute_ndsm(dsm_p, dtm_p, tmp / "ndsm.tif")

    bigger = write_raster(tmp_path / "bigger.tif",
                          np.full((N * 2, N * 2), 5.0, dtype="float32"))
    assert surface.is_stale(ndsm, bigger)


def test_is_stale_detects_a_shifted_origin(surfaces, tmp_path):
    """Same size, different place — still stale."""
    dsm_p, dtm_p, tmp = surfaces
    ndsm = surface.compute_ndsm(dsm_p, dtm_p, tmp / "ndsm.tif")

    shifted = tmp_path / "shifted.tif"
    prof = dict(driver="GTiff", height=N, width=N, count=1, dtype="float32",
                crs="EPSG:31370", nodata=surface.NODATA,
                transform=from_origin(X0 + 5000, Y0 + N, 1, 1))
    with rasterio.open(shifted, "w", **prof) as dst:
        dst.write(np.zeros((N, N), dtype="float32"), 1)
    assert surface.is_stale(ndsm, shifted)


def test_grid_of_returns_transform_and_shape(surfaces):
    dsm_p, _, _ = surfaces
    transform, w, h = surface.grid_of(dsm_p)
    assert (w, h) == (N, N)
    assert transform.a == 1.0
