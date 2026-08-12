"""Moon position and eclipse circumstances for a specific observer.

`solar.py` answers "where is the sun". This module answers "how much of it is
covered, and when did that start and stop", which needs the moon.

Why the moon needs more work than the sun
-----------------------------------------
The sun's position is smooth and a short series gets it to ~0.01 deg. The moon
is perturbed by the sun badly enough that a usable position needs a truncated
ELP series (Meeus, *Astronomical Algorithms* 2nd ed., ch. 47). More
importantly the moon is close, so it suffers **parallax**: an observer on the
surface sees it up to ~1 deg away from where a geocentric calculation puts it.
That is roughly twice the sun's diameter, so skipping the correction does not
shift contact times by seconds, it invalidates them.

Accuracy
--------
Position is good to a few arcseconds of the full ELP series; the limit here is
the truncated tables, not the method. Nutation (~17 arcsec) is neglected. The
sun-moon separation changes at ~0.5 deg/hour, so an 0.01 deg position error is
worth about a second of contact time. Cross-checked against published
predictions in notebook 01.

Conventions
-----------
Angles in degrees unless a name ends in `_rad`. Times are UTC hours on the
event date. Distances in km, except the sun's, which is in AU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .solar import (ANTWERP_LAT, ANTWERP_LON, EVENT_DAY, EVENT_MONTH,
                    EVENT_YEAR, julian_day)

EARTH_RADIUS_KM = 6378.14
MOON_RADIUS_KM = 1737.4
#: Sun's apparent radius at 1 AU, in arcseconds.
SUN_RADIUS_ARCSEC = 959.63
#: Antwerp's observer height for the parallax correction. The correction is
#: ~0.0001 deg per 100 m, so this only has to be roughly right.
ANTWERP_HEIGHT_M = 7.0

# --------------------------------------------------------------------------
# Meeus ch. 47 periodic terms.  (D, M, M', F, sigma_l [1e-6 deg], sigma_r [1e-3 km])
# Truncated after the terms that matter at our precision.
# --------------------------------------------------------------------------
_TERMS_LR = (
    (0, 0, 1, 0, 6288774, -20905355), (2, 0, -1, 0, 1274027, -3699111),
    (2, 0, 0, 0, 658314, -2955968),   (0, 0, 2, 0, 213618, -569925),
    (0, 1, 0, 0, -185116, 48888),     (0, 0, 0, 2, -114332, -3149),
    (2, 0, -2, 0, 58793, 246158),     (2, -1, -1, 0, 57066, -152138),
    (2, 0, 1, 0, 53322, -170733),     (2, -1, 0, 0, 45758, -204586),
    (0, 1, -1, 0, -40923, -129620),   (1, 0, 0, 0, -34720, 108743),
    (0, 1, 1, 0, -30383, 104755),     (2, 0, 0, -2, 15327, 10321),
    (0, 0, 1, 2, -12528, 0),          (0, 0, 1, -2, 10980, 79661),
    (4, 0, -1, 0, 10675, -34782),     (0, 0, 3, 0, 10034, -23210),
    (4, 0, -2, 0, 8548, -21636),      (2, 1, -1, 0, -7888, 24208),
    (2, 1, 0, 0, -6766, 30824),       (1, 0, -1, 0, -5163, -8379),
    (1, 1, 0, 0, 4987, -16675),       (2, -1, 1, 0, 4036, -12831),
    (2, 0, 2, 0, 3994, -10445),       (4, 0, 0, 0, 3861, -11650),
    (2, 0, -3, 0, 3665, 14403),       (0, 1, -2, 0, -2689, -7003),
    (2, 0, -1, 2, -2602, 0),          (2, -1, -2, 0, 2390, 10056),
    (1, 0, 1, 0, -2348, 6322),        (2, -2, 0, 0, 2236, -9884),
    (0, 1, 2, 0, -2120, 5751),        (0, 2, 0, 0, -2069, 0),
    (2, -2, -1, 0, 2048, -4950),      (2, 0, 1, -2, -1773, 4130),
    (2, 0, 0, 2, -1595, 0),           (4, -1, -1, 0, 1215, -3958),
    (0, 0, 2, 2, -1110, 0),           (3, 0, -1, 0, -892, 3258),
    (2, 1, 1, 0, -810, 2616),         (4, -1, -2, 0, 759, -1897),
    (0, 2, -1, 0, -713, -2117),       (2, 2, -1, 0, -700, 2354),
    (2, 1, -2, 0, 691, 0),            (2, -1, 0, -2, 596, 0),
    (4, 0, 1, 0, 549, -1423),         (0, 0, 4, 0, 537, -1117),
    (4, -1, 0, 0, 520, -1571),        (1, 0, -2, 0, -487, -1739),
    (2, 1, 0, -2, -399, 0),           (0, 0, 2, -2, -381, -4421),
    (1, 1, 1, 0, 351, 0),             (3, 0, -2, 0, -340, 0),
    (4, 0, -3, 0, 330, 0),            (2, -1, 2, 0, 327, 0),
    (0, 2, 1, 0, -323, 1165),         (1, 1, -1, 0, 299, 0),
    (2, 0, 3, 0, 294, 0),             (2, 0, -1, -2, 0, 8752),
)

#: (D, M, M', F, sigma_b [1e-6 deg])
_TERMS_B = (
    (0, 0, 0, 1, 5128122), (0, 0, 1, 1, 280602),  (0, 0, 1, -1, 277693),
    (2, 0, 0, -1, 173237), (2, 0, -1, 1, 55413),  (2, 0, -1, -1, 46271),
    (2, 0, 0, 1, 32573),   (0, 0, 2, 1, 17198),   (2, 0, 1, -1, 9266),
    (0, 0, 2, -1, 8822),   (2, -1, 0, -1, 8216),  (2, 0, -2, -1, 4324),
    (2, 0, 1, 1, 4200),    (2, 1, 0, -1, -3359),  (2, -1, -1, 1, 2463),
    (2, -1, 0, 1, 2211),   (2, -1, -1, -1, 2065), (0, 1, -1, -1, -1870),
    (4, 0, -1, -1, 1828),  (0, 1, 0, 1, -1794),   (0, 0, 0, 3, -1749),
    (0, 1, -1, 1, -1565),  (1, 0, 0, 1, -1491),   (0, 1, 1, 1, -1475),
    (0, 1, 1, -1, -1410),  (0, 1, 0, -1, -1344),  (1, 0, 0, -1, -1335),
    (0, 0, 3, 1, 1107),    (4, 0, 0, -1, 1021),   (4, 0, -1, 1, 833),
)


@dataclass(frozen=True)
class Circumstances:
    """Sun and moon as seen from one place at one instant."""
    separation_deg: float      #: angular distance between the two centres
    sun_radius_deg: float
    moon_radius_deg: float
    obscuration: float         #: fraction of the sun's AREA hidden, 0-1

    @property
    def in_eclipse(self) -> bool:
        return self.separation_deg < self.sun_radius_deg + self.moon_radius_deg


def _gmst_deg(jd: float) -> float:
    """Greenwich mean sidereal time in degrees."""
    n = jd - 2451545.0
    return ((18.697374558 + 24.06570982441908 * n) % 24) * 15.0


def _obliquity_rad(jd: float) -> float:
    return math.radians(23.439 - 0.0000004 * (jd - 2451545.0))


def _to_equatorial(lon_deg: float, lat_deg: float, obliquity_rad: float
                   ) -> tuple[float, float]:
    """Ecliptic longitude/latitude -> right ascension/declination, in degrees."""
    lam, beta = math.radians(lon_deg), math.radians(lat_deg)
    ra = math.atan2(
        math.sin(lam) * math.cos(obliquity_rad)
        - math.tan(beta) * math.sin(obliquity_rad),
        math.cos(lam),
    )
    dec = math.asin(
        math.sin(beta) * math.cos(obliquity_rad)
        + math.cos(beta) * math.sin(obliquity_rad) * math.sin(lam)
    )
    return math.degrees(ra) % 360, math.degrees(dec)


def moon_ecliptic(jd: float) -> tuple[float, float, float]:
    """Geocentric ecliptic longitude, latitude (deg) and distance (km).

    Meeus ch. 47. `E` corrects the terms involving the sun's mean anomaly for
    the slow decrease in Earth's orbital eccentricity; it enters once for
    |M| = 1 and twice for |M| = 2.
    """
    t = (jd - 2451545.0) / 36525.0

    lp = (218.3164477 + 481267.88123421 * t - 0.0015786 * t**2
          + t**3 / 538841 - t**4 / 65194000) % 360          # mean longitude
    d = (297.8501921 + 445267.1114034 * t - 0.0018819 * t**2
         + t**3 / 545868 - t**4 / 113065000) % 360          # mean elongation
    m = (357.5291092 + 35999.0502909 * t - 0.0001536 * t**2
         + t**3 / 24490000) % 360                           # sun mean anomaly
    mp = (134.9633964 + 477198.8675055 * t + 0.0087414 * t**2
          + t**3 / 69699 - t**4 / 14712000) % 360           # moon mean anomaly
    f = (93.2720950 + 483202.0175233 * t - 0.0036539 * t**2
         - t**3 / 3526000 + t**4 / 863310000) % 360         # argument of latitude

    e = 1 - 0.002516 * t - 0.0000074 * t**2

    sum_l = sum_r = sum_b = 0.0
    for cd, cm, cmp_, cf, cl, cr in _TERMS_LR:
        arg = math.radians(cd * d + cm * m + cmp_ * mp + cf * f)
        ecc = e ** abs(cm)
        sum_l += cl * ecc * math.sin(arg)
        sum_r += cr * ecc * math.cos(arg)
    for cd, cm, cmp_, cf, cb in _TERMS_B:
        arg = math.radians(cd * d + cm * m + cmp_ * mp + cf * f)
        sum_b += cb * (e ** abs(cm)) * math.sin(arg)

    # Additive terms: Venus (A1), Jupiter (A2) and the flattening of the Earth.
    a1 = math.radians((119.75 + 131.849 * t) % 360)
    a2 = math.radians((53.09 + 479264.290 * t) % 360)
    a3 = math.radians((313.45 + 481266.484 * t) % 360)
    sum_l += (3958 * math.sin(a1) + 1962 * math.sin(math.radians(lp - f))
              + 318 * math.sin(a2))
    sum_b += (-2235 * math.sin(math.radians(lp))
              + 382 * math.sin(a3)
              + 175 * math.sin(a1 - math.radians(f))
              + 175 * math.sin(a1 + math.radians(f))
              + 127 * math.sin(math.radians(lp - mp))
              - 115 * math.sin(math.radians(lp + mp)))

    return ((lp + sum_l / 1e6) % 360,
            sum_b / 1e6,
            385000.56 + sum_r / 1000.0)


def _topocentric(ra_deg: float, dec_deg: float, distance_km: float, jd: float,
                 lat: float, lon: float, height_m: float) -> tuple[float, float]:
    """Shift a geocentric RA/dec to what an observer on the surface sees.

    Meeus ch. 40. For the moon this is worth up to ~1 deg, so it is not
    optional; for the sun it is ~9 arcsec and would be.
    """
    sin_parallax = EARTH_RADIUS_KM / distance_km
    rlat = math.radians(lat)
    u = math.atan(0.99664719 * math.tan(rlat))
    rho_sin = 0.99664719 * math.sin(u) + (height_m / 6378140.0) * math.sin(rlat)
    rho_cos = math.cos(u) + (height_m / 6378140.0) * math.cos(rlat)

    hour_angle = math.radians((_gmst_deg(jd) + lon - ra_deg) % 360)
    dec = math.radians(dec_deg)

    d_ra = math.atan2(
        -rho_cos * sin_parallax * math.sin(hour_angle),
        math.cos(dec) - rho_cos * sin_parallax * math.cos(hour_angle),
    )
    dec_topo = math.atan2(
        (math.sin(dec) - rho_sin * sin_parallax) * math.cos(d_ra),
        math.cos(dec) - rho_cos * sin_parallax * math.cos(hour_angle),
    )
    return (ra_deg + math.degrees(d_ra)) % 360, math.degrees(dec_topo)


def _sun_equatorial(jd: float) -> tuple[float, float, float]:
    """Geocentric RA, dec (deg) and distance (AU), matching solar.sun_altaz."""
    n = jd - 2451545.0
    mean_longitude = (280.460 + 0.9856474 * n) % 360
    mean_anomaly = math.radians((357.528 + 0.9856003 * n) % 360)
    lam = (mean_longitude
           + 1.915 * math.sin(mean_anomaly)
           + 0.020 * math.sin(2 * mean_anomaly))
    radius_au = (1.00014 - 0.01671 * math.cos(mean_anomaly)
                 - 0.00014 * math.cos(2 * mean_anomaly))
    ra, dec = _to_equatorial(lam, 0.0, _obliquity_rad(jd))
    return ra, dec, radius_au


def angular_separation(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle angle between two equatorial positions, in degrees."""
    r1, d1 = math.radians(ra1), math.radians(dec1)
    r2, d2 = math.radians(ra2), math.radians(dec2)
    cos_sep = (math.sin(d1) * math.sin(d2)
               + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_sep))))


def obscuration(separation_deg: float, sun_radius_deg: float,
                moon_radius_deg: float) -> float:
    """Fraction of the sun's DISC AREA covered by the moon.

    Not the fraction of its diameter - the two differ a lot. At 89% area
    coverage the moon's edge has crossed about 96% of the sun's diameter.

    The overlap of two circles is a lens, whose area is the sum of the two
    circular segments either side of the chord where the discs cross.
    """
    d, r_s, r_m = separation_deg, sun_radius_deg, moon_radius_deg
    if d >= r_s + r_m:
        return 0.0
    if d <= r_m - r_s:
        return 1.0                       # sun entirely behind the moon
    if d <= r_s - r_m:
        return (r_m / r_s) ** 2          # moon entirely inside the sun

    alpha = math.acos(max(-1.0, min(1.0, (d*d + r_s*r_s - r_m*r_m) / (2*d*r_s))))
    beta = math.acos(max(-1.0, min(1.0, (d*d + r_m*r_m - r_s*r_s) / (2*d*r_m))))
    lens = (r_s*r_s * (alpha - math.sin(alpha) * math.cos(alpha))
            + r_m*r_m * (beta - math.sin(beta) * math.cos(beta)))
    return lens / (math.pi * r_s * r_s)


def _to_horizontal(ra_deg: float, dec_deg: float, jd: float,
                   lat: float, lon: float) -> tuple[float, float]:
    """Equatorial -> altitude/azimuth, geometric (no refraction).

    Deliberately a duplicate of the tail of `solar.sun_altaz`. Refactoring that
    function to share this one would change the sun's numbers by rounding
    alone, and every pinned test and published figure depends on them.
    """
    gmst = ((18.697374558 + 24.06570982441908 * (jd - 2451545.0)) % 24) * 15.0
    hour_angle = math.radians((gmst + lon - ra_deg) % 360)
    dec, rlat = math.radians(dec_deg), math.radians(lat)

    altitude = math.asin(math.sin(rlat) * math.sin(dec)
                         + math.cos(rlat) * math.cos(dec) * math.cos(hour_angle))
    azimuth = math.atan2(
        -math.sin(hour_angle),
        math.tan(dec) * math.cos(rlat) - math.sin(rlat) * math.cos(hour_angle),
    )
    return math.degrees(altitude), math.degrees(azimuth) % 360


def moon_altaz(hour_utc: float,
               lat: float = ANTWERP_LAT,
               lon: float = ANTWERP_LON,
               height_m: float = ANTWERP_HEIGHT_M,
               year: int = EVENT_YEAR,
               month: int = EVENT_MONTH,
               day: int = EVENT_DAY) -> tuple[float, float]:
    """Topocentric altitude and azimuth of the moon, in degrees, unrefracted."""
    jd = julian_day(year, month, day, hour_utc)
    lam, beta, dist = moon_ecliptic(jd)
    ra, dec = _to_equatorial(lam, beta, _obliquity_rad(jd))
    ra, dec = _topocentric(ra, dec, dist, jd, lat, lon, height_m)
    return _to_horizontal(ra, dec, jd, lat, lon)


def disc_offset(hour_utc: float, **kwargs) -> tuple[float, float]:
    """Where the moon sits relative to the sun, as seen in the sky.

    Returns (dx, dy) in degrees in the observer's frame: dx positive towards
    increasing azimuth (to the right when facing the sun), dy positive upwards.
    This is what a figure needs to put the bite on the correct side - a
    schematic crescent drawn on the wrong limb is the kind of error a reader
    in Antwerp will notice on the night.

    Refraction is left out of both bodies: it lifts them together by very
    nearly the same amount, so it cancels in the difference.
    """
    jd_kwargs = {k: v for k, v in kwargs.items()
                 if k in {"lat", "lon", "year", "month", "day"}}
    sun_alt, sun_az = _sun_altaz_geometric(hour_utc, **jd_kwargs)
    moon_alt, moon_az = moon_altaz(hour_utc, **kwargs)

    d_az = (moon_az - sun_az + 180) % 360 - 180
    return d_az * math.cos(math.radians(sun_alt)), moon_alt - sun_alt


def _sun_altaz_geometric(hour_utc: float,
                         lat: float = ANTWERP_LAT,
                         lon: float = ANTWERP_LON,
                         year: int = EVENT_YEAR,
                         month: int = EVENT_MONTH,
                         day: int = EVENT_DAY) -> tuple[float, float]:
    """Unrefracted solar alt/az, for differencing against the moon."""
    jd = julian_day(year, month, day, hour_utc)
    ra, dec, _ = _sun_equatorial(jd)
    return _to_horizontal(ra, dec, jd, lat, lon)


def circumstances(hour_utc: float,
                  lat: float = ANTWERP_LAT,
                  lon: float = ANTWERP_LON,
                  height_m: float = ANTWERP_HEIGHT_M,
                  year: int = EVENT_YEAR,
                  month: int = EVENT_MONTH,
                  day: int = EVENT_DAY) -> Circumstances:
    """Separation, apparent radii and obscuration at one instant."""
    jd = julian_day(year, month, day, hour_utc)

    lam, beta, dist_km = moon_ecliptic(jd)
    m_ra, m_dec = _to_equatorial(lam, beta, _obliquity_rad(jd))
    m_ra, m_dec = _topocentric(m_ra, m_dec, dist_km, jd, lat, lon, height_m)

    s_ra, s_dec, radius_au = _sun_equatorial(jd)

    sep = angular_separation(s_ra, s_dec, m_ra, m_dec)
    r_sun = (SUN_RADIUS_ARCSEC / radius_au) / 3600.0
    # The moon's apparent radius grows as it approaches; at perigee vs apogee
    # this varies by ~12%, which is the difference between an annular and a
    # total eclipse elsewhere on the track.
    r_moon = math.degrees(math.asin(MOON_RADIUS_KM / dist_km))

    return Circumstances(sep, r_sun, r_moon,
                         obscuration(sep, r_sun, r_moon))


def _bisect(lo: float, hi: float, predicate, tol_hours: float = 1e-5) -> float:
    """Smallest t in [lo, hi] where `predicate` flips from False to True."""
    while hi - lo > tol_hours:
        mid = (lo + hi) / 2
        if predicate(mid):
            hi = mid
        else:
            lo = mid
    return hi


def contacts(search_start_utc: float = 15.0,
             search_end_utc: float = 21.0,
             step_hours: float = 1 / 120,
             **kwargs) -> dict:
    """First contact, maximum and last contact, in UTC hours.

    Scans coarsely for the sign changes, then bisects. `maximum` is the instant
    of least separation, refined by golden-section search rather than taken
    from the scan grid.

    Returns UTC hours; add `solar.UTC_OFFSET_HOURS` for local time. Any of the
    three may be None if it falls outside the search window - for Antwerp the
    sun sets while the eclipse is still running, so C4 is real but happens
    below the horizon.
    """
    def sep(t: float) -> float:
        c = circumstances(t, **kwargs)
        return c.separation_deg - (c.sun_radius_deg + c.moon_radius_deg)

    ts, vals, t = [], [], search_start_utc
    while t <= search_end_utc:
        ts.append(t); vals.append(sep(t)); t += step_hours

    c1 = c4 = None
    for i in range(len(ts) - 1):
        if vals[i] > 0 >= vals[i + 1]:
            c1 = _bisect(ts[i], ts[i + 1], lambda x: sep(x) <= 0)
        elif vals[i] <= 0 < vals[i + 1]:
            c4 = _bisect(ts[i], ts[i + 1], lambda x: sep(x) > 0)

    lo = c1 if c1 is not None else search_start_utc
    hi = c4 if c4 is not None else search_end_utc
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    while b - a > 1e-6:
        x1, x2 = b - gr * (b - a), a + gr * (b - a)
        if circumstances(x1, **kwargs).separation_deg < \
           circumstances(x2, **kwargs).separation_deg:
            b = x2
        else:
            a = x1
    maximum = (a + b) / 2

    return {
        "first_contact_utc": c1,
        "maximum_utc": maximum,
        "last_contact_utc": c4,
        "max_obscuration": circumstances(maximum, **kwargs).obscuration,
    }
