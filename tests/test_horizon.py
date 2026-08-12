"""Tests for the horizon ray-cast.

The strategy is to build synthetic surfaces whose correct answer can be worked
out on paper — a wall of known height at a known distance subtends a known
angle — so the tests check the geometry rather than merely that the code runs.

Run with:  pytest -v
"""

import math

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


from eclipse_viewshed import horizon, solar, surface

# Observer sits at the east edge of a scene extending west, matching the real
# geometry: the sun is in the west-northwest.
N = 1200
X0, Y0 = 151000, 211000          # south-west corner of the raster
OBS_X, OBS_Y = X0 + 1100, Y0 + 600
GROUND = 5.0


def write(path, arr, dtype="float32", nodata=surface.NODATA):
    prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                dtype=dtype, crs="EPSG:31370", nodata=nodata,
                transform=from_origin(X0, Y0 + arr.shape[0], 1, 1))
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)
    return path


@pytest.fixture
def flat(tmp_path):
    """Featureless ground at 5 m TAW."""
    return write(tmp_path / "flat.tif", np.full((N, N), GROUND, dtype="float32"))


def wall_at(tmp_path, distance_m, height_m, name="wall.tif"):
    """A north-south wall `distance_m` due west of the observer."""
    arr = np.full((N, N), GROUND, dtype="float32")
    col = int(round(OBS_X - distance_m - X0))
    arr[:, col - 1:col + 2] = GROUND + height_m
    return write(tmp_path / name, arr)


# ------------------------------------------------------------- geometry bits

def test_curvature_drop_known_values():
    """d^2 (1-k) / 2R, with k = 0.13."""
    assert horizon.curvature_drop(1000) == pytest.approx(0.068, abs=0.005)
    assert horizon.curvature_drop(5000) == pytest.approx(1.71, abs=0.02)
    assert horizon.curvature_drop(10000) == pytest.approx(6.83, abs=0.05)


def test_curvature_drop_is_quadratic():
    assert horizon.curvature_drop(2000) == pytest.approx(
        4 * horizon.curvature_drop(1000), rel=1e-9)


def test_refraction_reduces_the_drop():
    """k = 0 is pure geometry; k = 0.13 must give a smaller drop."""
    assert horizon.curvature_drop(5000, k=0.13) < horizon.curvature_drop(5000, k=0.0)


def test_sample_distances_start_beyond_the_observer():
    """Self-occlusion guard: no sample may sit on the observer's own cell."""
    d = horizon.sample_distances(5000, near_m=20)
    assert d.min() >= 20
    assert d.max() <= 5000


def test_sample_distances_are_dense_near_and_coarse_far():
    d = horizon.sample_distances(10000, near_m=20)
    near = np.diff(d[d < horizon.NEAR_MAX_M])
    far = np.diff(d[d > horizon.NEAR_MAX_M])
    assert near.max() <= horizon.NEAR_STEP_M + 1e-9
    assert far.min() >= horizon.FAR_STEP_M - 1e-9


# ------------------------------------------------------------- eye elevation

def test_resolve_eye_z_ground_mode_uses_dtm(tmp_path, flat):
    dsm = write(tmp_path / "dsm.tif", np.full((N, N), GROUND + 20, dtype="float32"))
    z, base = horizon.resolve_eye_z(OBS_X, OBS_Y, dsm, flat,
                                    z_mode="ground", eye_height_m=1.6)
    assert base == pytest.approx(GROUND, abs=0.01)
    assert z == pytest.approx(GROUND + 1.6, abs=0.01)


def test_resolve_eye_z_surface_mode_uses_dsm(tmp_path, flat):
    """A rooftop observer: 'surface' must read the roof, not the ground."""
    roof = np.full((N, N), GROUND, dtype="float32")
    roof[500:700, 1000:1200] = GROUND + 60      # covers the observer
    dsm = write(tmp_path / "dsm.tif", roof)
    z, base = horizon.resolve_eye_z(OBS_X, OBS_Y, dsm, flat,
                                    z_mode="surface", eye_height_m=1.6)
    assert base == pytest.approx(GROUND + 60, abs=0.01)
    assert z == pytest.approx(GROUND + 61.6, abs=0.01)


def test_resolve_eye_z_absolute_mode_is_verbatim(tmp_path, flat):
    z, base = horizon.resolve_eye_z(OBS_X, OBS_Y, flat, flat,
                                    z_mode="absolute", abs_eye_z_taw=68.0)
    assert z == 68.0 and base == 68.0


def test_resolve_eye_z_absolute_requires_a_value(tmp_path, flat):
    with pytest.raises(ValueError, match="abs_eye_z_taw"):
        horizon.resolve_eye_z(OBS_X, OBS_Y, flat, flat, z_mode="absolute")


def test_resolve_eye_z_rejects_unknown_mode(tmp_path, flat):
    with pytest.raises(ValueError, match="unknown z_mode"):
        horizon.resolve_eye_z(OBS_X, OBS_Y, flat, flat, z_mode="rooftop")


def test_raised_eye_gives_a_ground_offset(tmp_path, flat):
    """Wandelterras encoding: deck height folded into eye_height_m."""
    z, base = horizon.resolve_eye_z(OBS_X, OBS_Y, flat, flat,
                                    z_mode="ground", eye_height_m=6.6)
    assert z - base == pytest.approx(6.6, abs=1e-6)


# ---------------------------------------------------------- horizon geometry

def test_flat_ground_horizon_is_negative(tmp_path, flat):
    """From 1.6 m up over flat ground, everything is below eye level."""
    p = horizon.compute_horizon(flat, OBS_X, OBS_Y, GROUND + 1.6, ray_m=800)
    assert p.horizon_deg.max() < 0


def test_wall_subtends_the_expected_angle(tmp_path, flat):
    """A 25 m wall 500 m west, eye at 1.6 m: atan((25 - 1.6 - drop)/500)."""
    dsm = wall_at(tmp_path, 500, 25)
    p = horizon.compute_horizon(dsm, OBS_X, OBS_Y, GROUND + 1.6, ray_m=900)

    drop = horizon.curvature_drop(500)
    expected = math.degrees(math.atan2(25 - 1.6 - drop, 500))
    assert p.horizon_deg.max() == pytest.approx(expected, abs=0.05)


def test_horizon_records_the_controlling_distance(tmp_path, flat):
    dsm = wall_at(tmp_path, 500, 25)
    p = horizon.compute_horizon(dsm, OBS_X, OBS_Y, GROUND + 1.6, ray_m=900)
    i = int(np.argmax(p.horizon_deg))
    assert p.distance_m[i] == pytest.approx(500, abs=6)
    assert p.height_taw[i] == pytest.approx(GROUND + 25, abs=0.01)


def test_nearer_wall_of_equal_height_subtends_more(tmp_path, flat):
    a = horizon.compute_horizon(wall_at(tmp_path, 300, 20, "a.tif"),
                                OBS_X, OBS_Y, GROUND + 1.6, ray_m=900)
    b = horizon.compute_horizon(wall_at(tmp_path, 700, 20, "b.tif"),
                                OBS_X, OBS_Y, GROUND + 1.6, ray_m=900)
    assert a.horizon_deg.max() > b.horizon_deg.max()


def test_raising_the_eye_lowers_the_horizon(tmp_path, flat):
    """The whole point of an elevated viewpoint."""
    dsm = wall_at(tmp_path, 500, 25)
    low = horizon.compute_horizon(dsm, OBS_X, OBS_Y, GROUND + 1.6, ray_m=900)
    high = horizon.compute_horizon(dsm, OBS_X, OBS_Y, GROUND + 20.0, ray_m=900)
    assert high.horizon_deg.max() < low.horizon_deg.max()


def test_taller_wall_subtends_more(tmp_path, flat):
    a = horizon.compute_horizon(wall_at(tmp_path, 500, 10, "a.tif"),
                                OBS_X, OBS_Y, GROUND + 1.6, ray_m=900)
    b = horizon.compute_horizon(wall_at(tmp_path, 500, 40, "b.tif"),
                                OBS_X, OBS_Y, GROUND + 1.6, ray_m=900)
    assert b.horizon_deg.max() > a.horizon_deg.max()


def test_curvature_is_actually_applied(tmp_path, flat):
    """An obstruction far away must be discounted by the curvature drop.

    Two identical walls, one at 500 m and one at 5000 m, scaled so flat-earth
    geometry would give the same angle. Curvature must make the distant one
    lower.
    """
    near = wall_at(tmp_path, 500, 50, "n.tif")
    p_near = horizon.compute_horizon(near, OBS_X, OBS_Y, GROUND + 1.6, ray_m=900)
    flat_angle = math.degrees(math.atan2(50 - 1.6, 500))
    assert p_near.horizon_deg.max() < flat_angle


def test_self_occlusion_is_avoided(tmp_path, flat):
    """A tall block containing the observer must not become its own horizon."""
    arr = np.full((N, N), GROUND, dtype="float32")
    r = N - int(OBS_Y - Y0)
    c = int(OBS_X - X0)
    arr[r - 5:r + 6, c - 5:c + 6] = GROUND + 60      # the observer's own roof
    dsm = write(tmp_path / "roof.tif", arr)

    p = horizon.compute_horizon(dsm, OBS_X, OBS_Y, GROUND + 61.6,
                                ray_m=900, near_m=20)
    assert p.horizon_deg.max() < 0, "the observer's own structure occluded them"


# ------------------------------------------------------- class exclusion

def test_excluded_class_does_not_obstruct(tmp_path, flat):
    """A spurious tall return over water must not set the horizon."""
    arr = np.full((N, N), GROUND, dtype="float32")
    col = int(OBS_X - 500 - X0)
    arr[:, col - 1:col + 2] = GROUND + 40
    dsm = write(tmp_path / "dsm.tif", arr)

    cls = np.full((N, N), surface.CLASS_GROUND, dtype="uint8")
    cls[:, col - 1:col + 2] = surface.CLASS_WATER
    classes = write(tmp_path / "cls.tif", cls, dtype="uint8", nodata=0)

    kept = horizon.compute_horizon(dsm, OBS_X, OBS_Y, GROUND + 1.6, ray_m=900)
    dropped = horizon.compute_horizon(dsm, OBS_X, OBS_Y, GROUND + 1.6, ray_m=900,
                                      classes_path=classes,
                                      exclude_classes=(surface.CLASS_WATER,))
    assert kept.horizon_deg.max() > 3.0
    assert dropped.horizon_deg.max() < 0


def test_controlling_class_is_recorded(tmp_path, flat):
    arr = np.full((N, N), GROUND, dtype="float32")
    col = int(OBS_X - 500 - X0)
    arr[:, col - 1:col + 2] = GROUND + 25
    dsm = write(tmp_path / "dsm.tif", arr)

    cls = np.full((N, N), surface.CLASS_GROUND, dtype="uint8")
    cls[:, col - 1:col + 2] = surface.CLASS_BUILDING
    classes = write(tmp_path / "cls.tif", cls, dtype="uint8", nodata=0)

    p = horizon.compute_horizon(dsm, OBS_X, OBS_Y, GROUND + 1.6, ray_m=900,
                                classes_path=classes)
    i = int(np.argmax(p.horizon_deg))
    assert p.class_code[i] == surface.CLASS_BUILDING


# ------------------------------------------------------------- visibility

def synthetic_profile(level_deg: float) -> horizon.HorizonProfile:
    az = np.arange(270.0, 297.01, 0.1)
    return horizon.HorizonProfile(
        azimuth_deg=az,
        horizon_deg=np.full(az.shape, level_deg),
        distance_m=np.full(az.shape, 500.0),
        height_taw=np.full(az.shape, 30.0),
        class_code=np.zeros(az.shape, dtype="uint8"),
        z_eye_taw=6.6, name="synthetic")


def test_visibility_falls_with_a_higher_horizon():
    track = solar.sun_track(step_seconds=30)
    times = []
    for level in (0.0, 3.0, 6.0, 9.0):
        v = horizon.visibility(synthetic_profile(level), track)
        times.append(v["last_visible_centre"])
    assert times == sorted(times, reverse=True), times


def test_visibility_at_maximum_matches_the_geometry():
    """Maximum eclipse is at 7.83 deg, so a 6 deg skyline clears it and a
    9 deg skyline does not."""
    track = solar.sun_track(step_seconds=30)
    assert horizon.visibility(synthetic_profile(6.0), track)["visible_at_max"]
    assert not horizon.visibility(synthetic_profile(9.0), track)["visible_at_max"]


def test_clearance_at_maximum_is_signed_correctly():
    track = solar.sun_track(step_seconds=30)
    v = horizon.visibility(synthetic_profile(6.0), track)
    assert v["clearance_at_max"] == pytest.approx(7.83 - 6.0, abs=0.1)


def test_upper_limb_outlasts_the_centre():
    """The disc is 0.53 deg across, so its top edge survives longer."""
    track = solar.sun_track(step_seconds=30)
    v = horizon.visibility(synthetic_profile(3.0), track)
    assert v["minutes_visible_upper_limb"] > v["minutes_visible_centre"]


def test_a_wall_of_sky_blocks_everything():
    track = solar.sun_track(step_seconds=30)
    v = horizon.visibility(synthetic_profile(30.0), track)
    assert v["minutes_visible_centre"] == 0.0
    assert v["last_visible_centre"] is None
    assert v["visible_at_max"] is False


def test_visibility_reports_the_eye_height():
    track = solar.sun_track(step_seconds=30)
    v = horizon.visibility(synthetic_profile(3.0), track)
    assert v["z_eye_taw"] == pytest.approx(6.6)


def test_controlling_classes_shares_sum_to_one():
    p = synthetic_profile(3.0)
    p.class_code[:100] = surface.CLASS_BUILDING
    p.class_code[100:] = surface.CLASS_VEGETATION
    df = horizon.controlling_classes(p, surface.CLASS_LABELS)
    assert df["bearings"].sum() == p.azimuth_deg.size
    assert set(df["class"]) == {"building", "vegetation"}


def test_profile_frame_roundtrip():
    p = synthetic_profile(2.0)
    df = p.to_frame()
    assert len(df) == p.azimuth_deg.size
    assert {"azimuth_deg", "horizon_deg", "distance_m",
            "height_taw", "class_code"} <= set(df.columns)
    assert p.at_azimuth(284.0) == pytest.approx(2.0)


# -------------------------------------------------- coverage boundary

def test_observer_outside_raster_raises_legibly(tmp_path, flat):
    """Regression: rasterio's own error is `WindowError: Intersection is
    empty`, which says nothing about the cause. An observer outside the
    acquired tiles must say so."""
    far_x, far_y = X0 + 50_000, Y0 + 50_000
    with pytest.raises(horizon.OutsideCoverage, match="outside"):
        horizon.resolve_eye_z(far_x, far_y, flat, flat, z_mode="ground")


def test_outside_coverage_message_names_the_fix(tmp_path, flat):
    try:
        horizon.resolve_eye_z(X0 - 5000, Y0 - 5000, flat, flat)
    except horizon.OutsideCoverage as exc:
        assert "notebook 02" in str(exc)
    else:
        pytest.fail("expected OutsideCoverage")


def test_compute_horizon_also_guards_coverage(tmp_path, flat):
    with pytest.raises(horizon.OutsideCoverage):
        horizon.compute_horizon(flat, X0 + 50_000, Y0 + 50_000, 10.0, ray_m=500)


def test_outside_coverage_is_a_valueerror():
    """So existing `except ValueError` handlers still catch it."""
    assert issubclass(horizon.OutsideCoverage, ValueError)


# ------------------------------------------- stale / misaligned classes

def test_classes_raster_smaller_than_dsm_still_works(tmp_path, flat):
    """Regression: re-running notebook 02 with an extra observer grows the DSM
    mosaic, leaving notebook 03's classes raster covering less ground on the
    same grid. That must degrade gracefully, not raise."""
    dsm = wall_at(tmp_path, 500, 25)

    # Coverage is measured over the analysed fan, not the whole raster, so the
    # classes raster has to fall short *of the fan* for this to mean anything.
    # The observer is at Y0+600 and the wedge (273-294 deg) runs north-west, so
    # the fan reaches roughly Y0+966. A classes raster stopping at Y0+800
    # therefore covers it only partly.
    top = 800
    cls = np.full((top, N), surface.CLASS_GROUND, dtype="uint8")
    prof = dict(driver="GTiff", height=top, width=N, count=1, dtype="uint8",
                crs="EPSG:31370", nodata=0,
                transform=from_origin(X0, Y0 + top, 1, 1))
    small = tmp_path / "small_classes.tif"
    with rasterio.open(small, "w", **prof) as dst:
        dst.write(cls, 1)

    p = horizon.compute_horizon(dsm, OBS_X, OBS_Y, GROUND + 1.6, ray_m=900,
                                classes_path=small,
                                exclude_classes=(surface.CLASS_WATER,))
    assert p.horizon_deg.max() > 2.0          # the wall still obstructs
    assert 0.0 < p.class_coverage < 1.0       # and the shortfall is reported


def test_full_coverage_reports_one(tmp_path, flat):
    dsm = wall_at(tmp_path, 500, 25)
    cls = np.full((N, N), surface.CLASS_GROUND, dtype="uint8")
    classes = write(tmp_path / "cls.tif", cls, dtype="uint8", nodata=0)
    p = horizon.compute_horizon(dsm, OBS_X, OBS_Y, GROUND + 1.6, ray_m=900,
                                classes_path=classes)
    assert p.class_coverage == pytest.approx(1.0, abs=1e-6)


def test_mismatched_resolution_is_rejected(tmp_path, flat):
    """A 10 m classes raster against a 1 m DSM is a real error, not a shortfall."""
    dsm = wall_at(tmp_path, 500, 25)
    coarse = np.full((120, 120), surface.CLASS_GROUND, dtype="uint8")
    prof = dict(driver="GTiff", height=120, width=120, count=1, dtype="uint8",
                crs="EPSG:31370", nodata=0,
                transform=from_origin(X0, Y0 + N, 10, 10))
    path = tmp_path / "coarse.tif"
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(coarse, 1)

    with pytest.raises(ValueError, match="resolution"):
        horizon.compute_horizon(dsm, OBS_X, OBS_Y, GROUND + 1.6, ray_m=900,
                                classes_path=path)


def test_uncovered_pixels_are_unclassified_not_excluded(tmp_path, flat):
    """Pixels the classes raster misses come back as 0, so exclude_classes
    cannot silently remove real obstructions."""
    dsm = wall_at(tmp_path, 500, 40)
    tiny = np.full((10, 10), surface.CLASS_WATER, dtype="uint8")
    prof = dict(driver="GTiff", height=10, width=10, count=1, dtype="uint8",
                crs="EPSG:31370", nodata=0,
                transform=from_origin(X0, Y0 + 10, 1, 1))
    path = tmp_path / "tiny.tif"
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(tiny, 1)

    p = horizon.compute_horizon(dsm, OBS_X, OBS_Y, GROUND + 1.6, ray_m=900,
                                classes_path=path,
                                exclude_classes=(surface.CLASS_WATER,))
    assert p.horizon_deg.max() > 3.0


# ------------------------------------------------ cross-section / evidence

def test_ray_profile_finds_the_wall(tmp_path, flat):
    """The cross-section must show the wall at the right range and height."""
    dsm = wall_at(tmp_path, 500, 25)
    prof = horizon.ray_profile(dsm, OBS_X, OBS_Y, GROUND + 1.6, 270.0, ray_m=900)

    near_wall = prof[(prof.distance_m > 490) & (prof.distance_m < 510)]
    assert near_wall["elevation_taw"].max() == pytest.approx(GROUND + 25, abs=0.1)

    open_ground = prof[(prof.distance_m > 100) & (prof.distance_m < 400)]
    assert open_ground["elevation_taw"].max() == pytest.approx(GROUND, abs=0.1)


def test_ray_profile_apparent_elevation_is_below_true(tmp_path, flat):
    """Curvature makes distant ground sit lower than its true elevation."""
    prof = horizon.ray_profile(flat, OBS_X, OBS_Y, GROUND + 1.6, 270.0, ray_m=900)
    far = prof.iloc[-1]
    assert far["apparent_taw"] < far["elevation_taw"]
    assert (far["elevation_taw"] - far["apparent_taw"]) == pytest.approx(
        horizon.curvature_drop(far["distance_m"]), abs=1e-6)


def test_ray_profile_angle_matches_compute_horizon(tmp_path, flat):
    """The cross-section and the ray-cast must agree, or the figure would be
    illustrating something other than the analysis."""
    dsm = wall_at(tmp_path, 500, 25)
    prof = horizon.ray_profile(dsm, OBS_X, OBS_Y, GROUND + 1.6, 280.0, ray_m=900)
    h = horizon.compute_horizon(dsm, OBS_X, OBS_Y, GROUND + 1.6, ray_m=900,
                                az_min=280.0, az_max=280.0, az_step=1.0)
    assert prof["angle_deg"].max() == pytest.approx(h.horizon_deg[0], abs=0.02)


def test_ray_profile_starts_beyond_the_observer(tmp_path, flat):
    prof = horizon.ray_profile(flat, OBS_X, OBS_Y, GROUND + 1.6, 280.0,
                               ray_m=900, near_m=20)
    assert prof["distance_m"].min() >= 20


def test_controlling_feature_reports_the_evidence(tmp_path, flat):
    """A clearance number alone is unfalsifiable; the object behind it is not."""
    dsm = wall_at(tmp_path, 500, 25)
    h = horizon.compute_horizon(dsm, OBS_X, OBS_Y, GROUND + 1.6, ray_m=900)
    ev = horizon.controlling_feature(h, 280.0)

    assert ev["distance_m"] == pytest.approx(500, abs=10)
    assert ev["height_taw"] == pytest.approx(GROUND + 25, abs=0.2)
    assert ev["height_above_eye_m"] == pytest.approx(25 - 1.6, abs=0.2)


def test_controlling_feature_picks_the_nearest_bearing():
    p = synthetic_profile(3.0)
    ev = horizon.controlling_feature(p, 284.04)
    assert ev["azimuth_deg"] == pytest.approx(284.0, abs=0.06)


# ------------------------------------------------------ placement checking

def make_scene(tmp_path, water_cols, building_at=None):
    """Ground, optional building, and a water channel, with a classes raster."""
    dtm = np.full((N, N), GROUND, dtype="float32")
    dsm = dtm.copy()
    cls = np.full((N, N), surface.CLASS_GROUND, dtype="uint8")

    cls[:, water_cols] = surface.CLASS_WATER

    if building_at is not None:
        col = int(OBS_X - building_at - X0)
        dsm[:, col - 5:col + 6] = GROUND + 20
        cls[:, col - 5:col + 6] = surface.CLASS_BUILDING

    return (write(tmp_path / "dsm.tif", dsm),
            write(tmp_path / "dtm.tif", dtm),
            write(tmp_path / "cls.tif", cls, dtype="uint8", nodata=0))


def test_bank_placement_is_near_water_and_clear(tmp_path):
    """A point on a bank: water a few metres away, nothing tall nearby."""
    col = int(OBS_X - 30 - X0)
    dsm, dtm, cls = make_scene(tmp_path, slice(col - 60, col))
    s = horizon.describe_surroundings(OBS_X, OBS_Y, dsm, dtm, cls)

    assert s["nearest_water_m"] < 50
    assert s["tallest_within_50m"] == pytest.approx(0.0, abs=0.2)
    assert s["water_within_100m_pct"] > 10


def test_misplaced_point_is_far_from_water_with_a_building_next_door(tmp_path):
    """The Sint-Andries / MAS failure mode, as a test."""
    dsm, dtm, cls = make_scene(tmp_path, slice(0, 30), building_at=30)
    s = horizon.describe_surroundings(OBS_X, OBS_Y, dsm, dtm, cls)

    assert s["nearest_obstruction_m"] < 60
    assert s["tallest_within_50m"] > 15
    # No water anywhere in the search radius. `None` is the signal, and it is
    # a stronger one than a large distance: this point is nowhere near a bank.
    assert s["nearest_water_m"] is None
    assert s["water_within_100m_pct"] == 0.0


def test_surroundings_reports_ground_elevation(tmp_path):
    dsm, dtm, cls = make_scene(tmp_path, slice(0, 30))
    s = horizon.describe_surroundings(OBS_X, OBS_Y, dsm, dtm, cls)
    assert s["ground_taw"] == pytest.approx(GROUND, abs=0.1)


def test_surroundings_without_classes_still_reports_heights(tmp_path):
    dsm, dtm, _ = make_scene(tmp_path, slice(0, 30), building_at=30)
    s = horizon.describe_surroundings(OBS_X, OBS_Y, dsm, dtm, classes_path=None)
    assert "tallest_within_50m" in s
    assert "nearest_water_m" not in s


def test_surroundings_handles_no_water_in_range(tmp_path):
    dsm, dtm, cls = make_scene(tmp_path, slice(0, 1))
    s = horizon.describe_surroundings(OBS_X, OBS_Y, dsm, dtm, cls, radius_m=100)
    assert s["nearest_water_m"] is None
    assert s["water_within_100m_pct"] == 0.0
