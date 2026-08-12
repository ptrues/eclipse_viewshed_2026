"""Tests for the AOI geometry and tile grid.

The reusability property of this project — adding an observation point costs
only the tiles nobody else needed — is a property of the tile grid, so it is
tested here rather than left as an assertion in a notebook.

Run with:  pytest -v
"""


import pytest


from eclipse_viewshed import aoi, solar

# Scheldekaaien at Zuid. Used throughout as a known point with a known
# Lambert 72 equivalent, so a broken PROJ install is caught immediately.
ZUID_LAT, ZUID_LON = 51.21193309444983, 4.388558346421922
ZUID_X, ZUID_Y = 151384, 211331


# ------------------------------------------------------------- projection

def test_zuid_anchor_projects_correctly():
    """Guards against a broken PROJ install, which fails silently."""
    x, y = aoi.to_lambert72(ZUID_LAT, ZUID_LON)
    assert x == pytest.approx(ZUID_X, abs=50)
    assert y == pytest.approx(ZUID_Y, abs=50)


def test_bounds_roundtrip_to_wgs84():
    b = aoi.Bounds(151000, 211000, 152000, 212000)
    lon0, lat0, lon1, lat1 = b.to_wgs84()
    assert 4.3 < lon0 < 4.5 and 51.1 < lat0 < 51.3
    assert lon1 > lon0 and lat1 > lat0


# ------------------------------------------------------------ fan bounds

def test_fan_extends_west_not_east():
    """The wedge is west-northwest, so the box must be lopsided.

    A symmetric box around the observer would roughly quadruple the download.
    """
    b = aoi.fan_bounds(ZUID_X, ZUID_Y, ray_m=10_000)
    west_extent = ZUID_X - b.xmin
    east_extent = b.xmax - ZUID_X
    assert west_extent > 9_000
    assert east_extent < 1_000


def test_fan_extends_north_not_south():
    """Azimuths across the wedge all have a northward component."""
    b = aoi.fan_bounds(ZUID_X, ZUID_Y, ray_m=10_000)
    assert b.ymax - ZUID_Y > 3_000
    assert ZUID_Y - b.ymin < 1_000


def test_fan_contains_observer():
    b = aoi.fan_bounds(ZUID_X, ZUID_Y)
    assert b.xmin < ZUID_X < b.xmax
    assert b.ymin < ZUID_Y < b.ymax


def test_fan_scales_with_ray_length():
    small = aoi.fan_bounds(ZUID_X, ZUID_Y, ray_m=5_000)
    large = aoi.fan_bounds(ZUID_X, ZUID_Y, ray_m=10_000)
    assert large.area_km2 > small.area_km2


def test_fan_captures_due_west_extremum():
    """A sector spanning 270 deg must reach a full ray length west, even
    though neither arc endpoint is due west."""
    b = aoi.fan_bounds(0, 0, ray_m=1_000, az_min=260, az_max=280, pad_m=0)
    assert b.xmin == pytest.approx(-1_000, abs=1)


# ------------------------------------------------------------- tile grid

def test_tile_key_snaps_down_to_grid():
    assert aoi.tile_key(ZUID_X, ZUID_Y, 1000) == (151000, 211000)
    assert aoi.tile_key(151000, 211000, 1000) == (151000, 211000)
    assert aoi.tile_key(151999.9, 211999.9, 1000) == (151000, 211000)


def test_tile_key_is_stable_across_observers():
    """The property the cache depends on: the same ground gives the same key
    regardless of who asked. Cutting tiles relative to each AOI would break
    this and make cached tiles non-interchangeable."""
    assert aoi.tile_key(151500, 211500) == aoi.tile_key(151500.4, 211500.9)


def test_tile_bounds_inverts_tile_key():
    key = aoi.tile_key(ZUID_X, ZUID_Y, 1000)
    b = aoi.tile_bounds(key, 1000)
    assert (b.xmin, b.ymin) == key
    assert b.width == 1000 and b.height == 1000
    assert aoi.tile_key(b.xmin + 1, b.ymin + 1, 1000) == key


def test_tile_name_encodes_grid_size():
    """Tiles cut at different grid sizes must not collide in one cache dir."""
    k = (151000, 211000)
    assert aoi.tile_name(k, 1000) != aoi.tile_name(k, 2000)


def test_tiles_covering_is_complete():
    """Every corner of the bounds must fall inside some returned tile."""
    b = aoi.Bounds(151200, 211200, 153400, 212600)
    keys = aoi.tiles_covering(b, 1000)
    for x, y in [(b.xmin, b.ymin), (b.xmax, b.ymax),
                 (b.xmin, b.ymax), (b.xmax, b.ymin)]:
        assert aoi.tile_key(x, y, 1000) in keys


def test_tiles_covering_has_no_duplicates():
    keys = aoi.tiles_covering(aoi.Bounds(151000, 211000, 155000, 214000), 1000)
    assert len(keys) == len(set(keys))


def test_tiles_covering_count_is_exact():
    b = aoi.Bounds(151000, 211000, 154000, 213000)     # 3 x 2 km
    assert len(aoi.tiles_covering(b, 1000)) == 4 * 3   # inclusive of edges


# ------------------------------------------- the reusability property

# Real observation points from data/external/observers.csv, so these tests
# measure the configuration actually in use rather than an invented pair.
# Distances are from Scheldekaaien Zuid, the anchor above.
NIEUW_ZUID = (51.20501433632554, 4.3761034790145645)      # 1.16 km
WANDELTERRAS = (51.22104280425928, 4.395803746709888)     # 1.13 km
MAS = (51.22896510990676, 4.4047312094219775)             # 2.21 km
DROOGDOK = (51.2365, 4.4010)                              # 2.87 km


def test_nearby_viewpoint_reuses_most_tiles():
    """The primary use case: several points along the same stretch of bank.

    All look west-northwest, so fans a few hundred metres apart overlap almost
    entirely. Threshold set well below the observed ~76% so that ordinary
    coordinate corrections do not break the test — it asserts the property, not
    a tuned number.
    """
    zuid = set(aoi.tiles_for_observer(ZUID_LAT, ZUID_LON))
    nearby = set(aoi.tiles_for_observer(*NIEUW_ZUID))

    new_tiles = nearby - zuid
    reuse = 1 - len(new_tiles) / len(nearby)

    assert reuse > 0.6, f"only {reuse:.0%} reuse for a 1.2 km move"


def test_distant_viewpoint_still_reuses_substantially():
    """Reuse falls off with separation but must not vanish.

    Droogdokkenpark is ~2.9 km north of Zuid and reuses roughly 45%. Asserted
    loosely: the point is that a further viewpoint costs less than a fresh
    download, not that it costs any particular amount.
    """
    zuid = set(aoi.tiles_for_observer(ZUID_LAT, ZUID_LON))
    distant = set(aoi.tiles_for_observer(*DROOGDOK))

    new_tiles = distant - zuid
    reuse = 1 - len(new_tiles) / len(distant)

    assert reuse > 0.25, f"only {reuse:.0%} reuse; caching is not paying off"
    assert len(new_tiles) < len(distant), "a cached neighbour saved nothing"


def test_reuse_decreases_with_separation():
    """Sanity check on the geometry: the further apart, the less shared.

    Nieuw Zuid 1.16 km, MAS 2.21 km, Droogdokkenpark 2.87 km from the anchor.
    """
    zuid = set(aoi.tiles_for_observer(ZUID_LAT, ZUID_LON))

    def reuse(point):
        t = set(aoi.tiles_for_observer(*point))
        return 1 - len(t - zuid) / len(t)

    assert reuse(NIEUW_ZUID) > reuse(MAS) > reuse(DROOGDOK)


def test_union_is_smaller_than_sum():
    points = [(ZUID_LAT, ZUID_LON), DROOGDOK, NIEUW_ZUID]
    union = aoi.tiles_for_observers(points)
    total = sum(len(aoi.tiles_for_observer(la, lo)) for la, lo in points)
    assert len(union) < total


def test_union_contains_every_observers_tiles():
    points = [(ZUID_LAT, ZUID_LON), DROOGDOK]
    union = set(aoi.tiles_for_observers(points))
    for la, lo in points:
        assert set(aoi.tiles_for_observer(la, lo)) <= union


def test_union_is_deduplicated_and_sorted():
    points = [(ZUID_LAT, ZUID_LON), (ZUID_LAT, ZUID_LON)]
    union = aoi.tiles_for_observers(points)
    assert len(union) == len(set(union))
    assert union == sorted(union, key=lambda k: (k[1], k[0]))


def test_observer_tile_count_is_reasonable():
    """Sanity bound: one observer at 10 km should need tens of tiles, not
    thousands. Catches a units error or a runaway ray length."""
    n = len(aoi.tiles_for_observer(ZUID_LAT, ZUID_LON, ray_m=10_000))
    assert 20 < n < 200


# ------------------------------- regression: runtime-resolved configuration

def test_tile_m_is_resolved_at_call_time(monkeypatch):
    """Reassigning aoi.TILE_M must actually change behaviour.

    Regression test. These functions originally declared `tile_m: int = TILE_M`
    in their signatures. Python evaluates default arguments once, at function
    definition, so the value was frozen at import and `aoi.TILE_M = 500` was
    silently ignored — while notebook 02 documented that very assignment as the
    knob for shrinking WCS requests.
    """
    monkeypatch.setattr(aoi, "TILE_M", 500)

    assert aoi.tile_name((151000, 211000)).endswith("_t500")
    assert aoi.tile_bounds((151000, 211000)).width == 500
    assert aoi.tile_key(151750, 211750) == (151500, 211500)

    # A 2 x 1 km box holds 4x more 500 m tiles than 1000 m tiles.
    box = aoi.Bounds(151000, 211000, 153000, 212000)
    monkeypatch.setattr(aoi, "TILE_M", 1000)
    coarse = len(aoi.tiles_covering(box))
    monkeypatch.setattr(aoi, "TILE_M", 500)
    fine = len(aoi.tiles_covering(box))
    assert fine > coarse


def test_default_ray_m_is_resolved_at_call_time(monkeypatch):
    """Same failure mode on the ray length."""
    monkeypatch.setattr(aoi, "DEFAULT_RAY_M", 1_000)
    small = aoi.fan_bounds(ZUID_X, ZUID_Y)
    monkeypatch.setattr(aoi, "DEFAULT_RAY_M", 10_000)
    large = aoi.fan_bounds(ZUID_X, ZUID_Y)
    assert large.area_km2 > small.area_km2 * 10


def test_explicit_argument_still_overrides_module_default(monkeypatch):
    monkeypatch.setattr(aoi, "TILE_M", 1000)
    assert aoi.tile_name((151000, 211000), 250).endswith("_t250")
    assert aoi.tile_bounds((151000, 211000), 250).width == 250


# ------------------------------------------------- bearing / wedge selection

def test_bearing_cardinal_directions():
    """Azimuth is clockwise from north: N=0, E=90, S=180, W=270."""
    import numpy as np
    for (dx, dy), expected in [((0, 100), 0), ((100, 0), 90),
                               ((0, -100), 180), ((-100, 0), 270)]:
        az, _ = aoi.bearing_distance(0, 0, np.array([dx]), np.array([dy]))
        assert az[0] == pytest.approx(expected, abs=1e-6)


def test_bearing_matches_sun_azimuth_convention():
    """A point due WNW of the observer must report ~284 deg, the sun's azimuth
    at maximum eclipse — the two conventions have to agree or the wedge selects
    the wrong ground."""
    import math
    import numpy as np
    d = 1000.0
    x = math.sin(math.radians(284.0)) * d
    y = math.cos(math.radians(284.0)) * d
    az, dist = aoi.bearing_distance(0, 0, np.array([x]), np.array([y]))
    assert az[0] == pytest.approx(284.0, abs=1e-6)
    assert dist[0] == pytest.approx(d, abs=1e-6)


def test_bearing_broadcasts_row_and_column():
    import numpy as np
    xs = np.arange(5)[np.newaxis, :]
    ys = np.arange(3)[:, np.newaxis]
    az, dist = aoi.bearing_distance(0, 0, xs, ys)
    assert az.shape == (3, 5)
    assert dist.shape == (3, 5)


def test_in_wedge_selects_only_the_sector():
    """Derived from the constants, not hardcoded.

    An earlier version pinned 294 as inside and 295 as outside. When the wedge
    was widened to clear the setting upper limb, the test failed for the right
    reason but with a number that told you nothing.
    """
    import numpy as np
    lo, hi = solar.WEDGE_AZ_MIN, solar.WEDGE_AZ_MAX
    az = np.array([lo - 1, lo + 1, (lo + hi) / 2, hi - 1, hi + 1])
    dist = np.full(az.shape, 500.0)
    m = aoi.in_wedge(az, dist, max_distance=1000)
    assert list(m) == [False, True, True, True, False]


def test_in_wedge_respects_distance_limits():
    import numpy as np
    az = np.full(3, 284.0)
    dist = np.array([5.0, 500.0, 5000.0])
    m = aoi.in_wedge(az, dist, max_distance=1000, min_distance=50)
    assert list(m) == [False, True, False]


def test_to_wgs84_roundtrips_with_to_lambert72():
    """Argument orders differ deliberately: to_lambert72(lat, lon) matches
    observers.csv, to_wgs84 returns (lon, lat) matching pyproj always_xy."""
    x, y = aoi.to_lambert72(ZUID_LAT, ZUID_LON)
    lon, lat = aoi.to_wgs84(x, y)
    assert lat == pytest.approx(ZUID_LAT, abs=1e-7)
    assert lon == pytest.approx(ZUID_LON, abs=1e-7)


def test_module_to_wgs84_is_not_the_bounds_method():
    """Regression: an existence check for 'def to_wgs84(' matched the Bounds
    METHOD, so the module-level function silently never got added."""
    assert callable(aoi.to_wgs84)
    lon, lat = aoi.to_wgs84(151384, 211331)
    assert isinstance(lon, float) and isinstance(lat, float)
    assert 4.0 < lon < 5.0 and 51.0 < lat < 52.0
