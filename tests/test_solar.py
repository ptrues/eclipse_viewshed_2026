"""Tests for the solar geometry.

These are the guardrails. If the sun's position is wrong, every downstream
result is wrong in a way that looks entirely plausible, so it is worth pinning
the geometry against values that can be checked independently.

Run with:  pytest -v
"""

import math

import pytest


from eclipse_viewshed import solar


# --------------------------------------------------------------- julian day

def test_j2000_epoch():
    """J2000.0 is defined as 2000-01-01 12:00 UTC = JD 2451545.0 exactly."""
    assert solar.julian_day(2000, 1, 1, 12.0) == pytest.approx(2451545.0)


def test_julian_day_advances_by_one_per_day():
    a = solar.julian_day(2026, 8, 12, 0.0)
    b = solar.julian_day(2026, 8, 13, 0.0)
    assert b - a == pytest.approx(1.0)


def test_january_rollover():
    """Jan/Feb are remapped to months 13/14 of the prior year; check no jump."""
    dec = solar.julian_day(2025, 12, 31, 0.0)
    jan = solar.julian_day(2026, 1, 1, 0.0)
    assert jan - dec == pytest.approx(1.0)


# -------------------------------------------------------------- refraction

def test_refraction_larger_near_horizon():
    """Refraction grows sharply as the sun approaches the horizon."""
    assert solar.refraction_bennett(0.0) > solar.refraction_bennett(8.0)
    assert solar.refraction_bennett(8.0) > solar.refraction_bennett(45.0)


def test_refraction_magnitudes():
    """Published values: ~0.55 deg at the horizon, ~0.1 deg at 8 deg."""
    assert solar.refraction_bennett(0.0) == pytest.approx(0.55, abs=0.08)
    assert solar.refraction_bennett(8.0) == pytest.approx(0.10, abs=0.03)
    assert solar.refraction_bennett(45.0) == pytest.approx(0.017, abs=0.01)


# ---------------------------------------------------------- solar position

def test_azimuth_due_south_at_solar_noon():
    """In the northern hemisphere the sun peaks due south (az 180)."""
    best_alt, best_az = -90.0, None
    for i in range(240):                      # scan 10:00-14:00 UTC
        alt, az = solar.sun_altaz(10.0 + i / 60.0)
        if alt > best_alt:
            best_alt, best_az = alt, az
    assert best_az == pytest.approx(180.0, abs=1.0)


def test_sun_sets_in_the_northwest_in_august():
    """Well after the equinox, sunset is north of due west."""
    alt, az = solar.sun_altaz(19.0)           # 21:00 CEST, near sunset
    assert alt == pytest.approx(0.0, abs=1.5)
    assert 290.0 < az < 300.0


def test_maximum_eclipse_geometry():
    """Cross-check against published values for Antwerp.

    timeanddate.com gives maximum eclipse ~20:12 CEST with the sun ~8 deg up
    in the WNW. Tolerances are loose enough to allow for the low-precision
    algorithm but tight enough to catch a real error.
    """
    alt, az = solar.sun_altaz(18.2)           # 20:12 CEST
    assert alt == pytest.approx(7.8, abs=0.5)
    assert az == pytest.approx(284.0, abs=1.0)


def test_refraction_raises_apparent_altitude():
    true_alt, _ = solar.sun_altaz(18.2, apply_refraction=False)
    app_alt, _ = solar.sun_altaz(18.2, apply_refraction=True)
    assert app_alt > true_alt
    assert app_alt - true_alt == pytest.approx(0.1, abs=0.05)


# --------------------------------------------------------------- sun track

def test_track_spans_the_wedge():
    """The sun should sweep through the azimuth wedge we analyse."""
    track = solar.sun_track()
    assert track["azimuth_deg"].min() < solar.WEDGE_AZ_MIN
    assert track["azimuth_deg"].max() > solar.WEDGE_AZ_MAX


def test_track_is_monotonically_descending():
    """Evening: altitude only decreases, azimuth only increases."""
    track = solar.sun_track()
    assert (track["altitude_deg"].diff().dropna() < 0).all()
    assert (track["azimuth_deg"].diff().dropna() > 0).all()


def test_at_time_lookup():
    track = solar.sun_track()
    row = solar.at_time(track, 20.2)
    assert row["cest"] == "20:12"


# ------------------------------------------------------- blocking geometry

def test_blocking_height_scales_with_distance():
    """The core intuition of the project: the far field barely matters."""
    assert solar.blocking_height(100, 8.0) == pytest.approx(14.1, abs=0.5)
    assert solar.blocking_height(500, 8.0) == pytest.approx(70.3, abs=1.0)
    assert solar.blocking_height(5000, 8.0) == pytest.approx(702.6, abs=5.0)


def test_far_bank_trees_cannot_block_maximum():
    """A 25 m poplar at 500 m subtends well under the 7.7 deg maximum."""
    angle = math.degrees(math.atan2(25 - 1.6, 500))
    assert angle < 3.0


def test_far_bank_tower_can_block_maximum():
    """An 80 m tower at 500 m does exceed it."""
    angle = math.degrees(math.atan2(80 - 1.6, 500))
    assert angle > 7.7


# ------------------------------------------------------- clock formatting

def test_format_cest_basic():
    assert solar.format_cest(20.0) == "20:00"
    assert solar.format_cest(20.5) == "20:30"
    assert solar.format_cest(9.25) == "09:15"


def test_format_cest_carries_into_the_next_hour():
    """Regression. The original formulation rounded the minutes to 60, wrapped
    them to 00 with `% 60`, and never incremented the hour — so 20:59:45
    printed as '20:00', a full hour early. It corrupted exactly the last-visible
    times, which cluster near the top of the hour."""
    assert solar.format_cest(20 + 59.75 / 60) == "21:00"
    assert solar.format_cest(19 + 59.9 / 60) == "20:00"
    assert solar.format_cest(20.9958333) == "21:00"


def test_format_cest_wraps_past_midnight():
    assert solar.format_cest(23 + 59.9 / 60) == "00:00"
    assert solar.format_cest(24.0) == "00:00"


def test_track_times_increase_monotonically():
    """The carry bug made the printed clock jump backwards mid-track."""
    track = solar.sun_track(step_seconds=15)
    times = track["cest"].tolist()
    as_minutes = [int(t[:2]) * 60 + int(t[3:]) for t in times]
    assert as_minutes == sorted(as_minutes), "printed times are not monotonic"


def test_track_time_matches_elapsed_samples():
    """End-to-end: the last sample's clock time must equal start + duration."""
    track = solar.sun_track(start_cest=19.0, end_cest=21.25, step_seconds=15)
    assert track.iloc[0]["cest"] == "19:00"
    assert track.iloc[-1]["cest"] == "21:15"
