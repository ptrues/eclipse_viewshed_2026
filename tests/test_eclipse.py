"""Moon position, obscuration and contact times.

The published circumstances for Antwerp are the reference: first contact
19:18, maximum 20:13, ~89.2% obscuration (timeanddate.com). Agreement to a
minute is the bar - the truncated lunar series and the low-precision sun are
each worth a few seconds of contact time.
"""

import pytest


from eclipse_viewshed import eclipse, solar

CEST = solar.UTC_OFFSET_HOURS


@pytest.fixture(scope="module")
def contacts():
    return eclipse.contacts()


def test_moon_distance_is_physical():
    jd = solar.julian_day(2026, 8, 12, 18.2)
    _, beta, dist = eclipse.moon_ecliptic(jd)
    assert 356_000 < dist < 407_000        # perigee to apogee
    assert abs(beta) < 6.0                 # never far off the ecliptic


def test_parallax_moves_the_moon_by_about_a_degree():
    """Skipping the topocentric correction is a ~1 deg error, not a rounding one."""
    jd = solar.julian_day(2026, 8, 12, 18.2)
    lam, beta, dist = eclipse.moon_ecliptic(jd)
    ra, dec = eclipse._to_equatorial(lam, beta, eclipse._obliquity_rad(jd))
    ra_t, dec_t = eclipse._topocentric(ra, dec, dist, jd,
                                       solar.ANTWERP_LAT, solar.ANTWERP_LON, 7.0)
    shift = eclipse.angular_separation(ra, dec, ra_t, dec_t)
    assert 0.3 < shift < 1.1


def test_obscuration_endpoints():
    assert eclipse.obscuration(1.0, 0.26, 0.27) == 0.0        # discs apart
    assert eclipse.obscuration(0.0, 0.26, 0.27) == 1.0        # total
    assert eclipse.obscuration(0.0, 0.30, 0.15) == pytest.approx(0.25)  # annular


def test_obscuration_is_area_not_diameter():
    """Half the diameter covered is far less than half the area."""
    r = 0.26
    half_diameter = eclipse.obscuration(r, r, r)
    assert half_diameter < 0.42


def test_obscuration_decreases_with_separation():
    r_s, r_m = 0.2631, 0.2712
    vals = [eclipse.obscuration(d, r_s, r_m) for d in (0.0, 0.1, 0.2, 0.4, 0.6)]
    assert vals == sorted(vals, reverse=True)


def test_contact_times_match_published(contacts):
    first = (contacts["first_contact_utc"] + CEST) * 60
    maximum = (contacts["maximum_utc"] + CEST) * 60
    last = (contacts["last_contact_utc"] + CEST) * 60
    assert abs(first - (19 * 60 + 18)) <= 2.0
    assert abs(maximum - (20 * 60 + 13)) <= 2.0
    assert abs(last - (21 * 60 + 5)) <= 3.0


def test_obscuration_matches_published(contacts):
    assert contacts["max_obscuration"] == pytest.approx(0.892, abs=0.01)


def test_ordering(contacts):
    assert (contacts["first_contact_utc"]
            < contacts["maximum_utc"]
            < contacts["last_contact_utc"])


def test_eclipse_is_in_progress_only_inside_the_contacts(contacts):
    c1, c4 = contacts["first_contact_utc"], contacts["last_contact_utc"]
    assert not eclipse.circumstances(c1 - 0.05).in_eclipse
    assert eclipse.circumstances((c1 + c4) / 2).in_eclipse
    assert not eclipse.circumstances(c4 + 0.05).in_eclipse


def test_sun_still_up_at_maximum(contacts):
    altitude, azimuth = solar.sun_altaz(contacts["maximum_utc"])
    assert 6.0 < altitude < 9.0
    assert solar.WEDGE_AZ_MIN < azimuth < solar.WEDGE_AZ_MAX


def test_moon_is_bigger_than_the_sun_here(contacts):
    """This eclipse is total somewhere, so the moon must over-cover the disc."""
    c = eclipse.circumstances(contacts["maximum_utc"])
    assert c.moon_radius_deg > c.sun_radius_deg
