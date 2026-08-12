"""
Solar position for the 12 August 2026 partial eclipse over Antwerp.

Why hand-roll this instead of just calling pvlib?

Two reasons. First, it is the most important input to the project - if the sun's
azimuth is wrong by two degrees, every downstream conclusion is wrong - so it is
worth understanding rather than treating as a black box. Second, an independent
implementation allows cross-validation against pvlib in notebook 01.

The algorithm is the low-precision solar position from the Astronomical
Almanac, accurate to roughly 0.01 deg over 1950-2050. That is far better than
we need: our surface model has metre-scale error, which at 500 m subtends
~0.1 deg. The ephemeris is not the weak link.

Vertical datum note: nothing here touches elevation. Heights elsewhere in this
project are metres TAW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Site and event constants
# --------------------------------------------------------------------------

# A single city-centre reference for the sun's position - deliberately NOT tied
# to any observation point. Solar position varies so slowly that moving 10 km
# changes the azimuth by well under 0.1 deg, so one reference serves every
# viewpoint.
ANTWERP_LAT = 51.2194
ANTWERP_LON = 4.4025

# CEST is UTC+2 in August.
UTC_OFFSET_HOURS = 2

EVENT_YEAR, EVENT_MONTH, EVENT_DAY = 2026, 8, 12

# The azimuth wedge the eclipsed sun traverses. The lower bound sits just west
# of first contact (274.06 deg); the upper bound has to clear the moment the
# sun's UPPER LIMB sets, at 295.27 deg, not last contact at 294.39 - otherwise
# an observer high enough to see past the skyline has their visible-minutes cut
# short by the edge of the analysis rather than by anything real. 296 leaves
# margin on both.
# Everything outside this is irrelevant to the analysis.
WEDGE_AZ_MIN = 273.0
WEDGE_AZ_MAX = 296.0


@dataclass(frozen=True)
class SunPosition:
    """Apparent solar position at an instant."""
    utc_hours: float
    cest: str
    altitude_deg: float   # refraction-corrected (apparent)
    azimuth_deg: float    # degrees clockwise from true north


# --------------------------------------------------------------------------
# Core geometry
# --------------------------------------------------------------------------

def julian_day(year: int, month: int, day: int, hour_utc: float) -> float:
    """Convert a calendar date and UTC hour to a Julian Day Number.

    Julian Day is a continuous count of days since 4713 BC, which turns
    calendar arithmetic (leap years, month lengths) into ordinary subtraction.
    Every solar position algorithm starts here.

    January and February are treated as months 13 and 14 of the previous year,
    a bookkeeping trick that puts the leap day at the end of the "year" so the
    365.25 term works cleanly.
    """
    if month <= 2:
        year -= 1
        month += 12

    century = year // 100
    # Gregorian calendar correction (skipped leap centuries).
    gregorian = 2 - century + century // 4

    return (int(365.25 * (year + 4716))
            + int(30.6001 * (month + 1))
            + day + gregorian - 1524.5
            + hour_utc / 24.0)


def refraction_bennett(true_altitude_deg: float) -> float:
    """Atmospheric refraction in degrees, to be ADDED to true altitude.

    The atmosphere bends light downward, so the sun appears higher than it
    geometrically is. This matters here: the correction is ~0.1 deg at 8 deg
    altitude and ~0.5 deg at the horizon - comparable to the precision we are
    trying to resolve. At sunset the sun's whole disc (0.53 deg) is already
    geometrically below the horizon.

    Bennett (1982), the standard practical formula.
    """
    if true_altitude_deg < -1.0:
        return 0.0
    denom = true_altitude_deg + 10.3 / (true_altitude_deg + 5.11)
    return (1.02 / math.tan(math.radians(denom))) / 60.0


def format_cest(local_hours: float) -> str:
    """Decimal local hours as HH:MM, carrying correctly into the next hour.

    The obvious formulation is wrong:

        f"{int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}"

    At 20.9958 hours (20:59:45) the minutes round to 60, `% 60` wraps them
    back to 00, and the hour is never incremented — so 20:59:45 prints as
    "20:00", an hour early. Roughly one sample in 240 is affected at 15-second
    steps, and they cluster at the top of the hour, which is exactly where the
    last-visible times fall.

    Rounding to whole minutes first and then splitting makes the carry
    automatic.
    """
    total_minutes = round(local_hours * 60)
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours % 24:02d}:{minutes:02d}"


def sun_altaz(hour_utc: float,
              lat: float = ANTWERP_LAT,
              lon: float = ANTWERP_LON,
              year: int = EVENT_YEAR,
              month: int = EVENT_MONTH,
              day: int = EVENT_DAY,
              apply_refraction: bool = True) -> tuple[float, float]:
    """Apparent solar altitude and azimuth in degrees.

    Azimuth is measured clockwise from true north (N=0, E=90, S=180, W=270),
    which is the convention our horizon rays use.

    Walks through: mean longitude -> true ecliptic longitude -> equatorial
    coordinates (RA/dec) -> hour angle -> horizontal coordinates (alt/az).
    """
    jd = julian_day(year, month, day, hour_utc)
    n = jd - 2451545.0  # days since the J2000.0 epoch

    # --- Sun's position on the ecliptic ---
    # Mean longitude: where the sun would be if Earth's orbit were circular.
    mean_longitude = (280.460 + 0.9856474 * n) % 360
    # Mean anomaly: angular position in the elliptical orbit.
    mean_anomaly = math.radians((357.528 + 0.9856003 * n) % 360)
    # Equation of centre corrects the circular assumption to the real ellipse.
    ecliptic_longitude = math.radians(
        mean_longitude
        + 1.915 * math.sin(mean_anomaly)
        + 0.020 * math.sin(2 * mean_anomaly)
    )
    # Obliquity: Earth's axial tilt, drifting very slowly.
    obliquity = math.radians(23.439 - 0.0000004 * n)

    # --- Ecliptic -> equatorial ---
    right_ascension = math.atan2(
        math.cos(obliquity) * math.sin(ecliptic_longitude),
        math.cos(ecliptic_longitude),
    )
    declination = math.asin(math.sin(obliquity) * math.sin(ecliptic_longitude))

    # --- Equatorial -> horizontal (observer-dependent) ---
    # Local sidereal time: where the observer's meridian points among the stars.
    gmst = (18.697374558 + 24.06570982441908 * n) % 24
    local_sidereal = math.radians((gmst * 15 + lon) % 360)
    hour_angle = local_sidereal - right_ascension

    rlat = math.radians(lat)
    altitude = math.asin(
        math.sin(rlat) * math.sin(declination)
        + math.cos(rlat) * math.cos(declination) * math.cos(hour_angle)
    )
    azimuth = math.atan2(
        -math.sin(hour_angle),
        math.tan(declination) * math.cos(rlat)
        - math.sin(rlat) * math.cos(hour_angle),
    )

    altitude_deg = math.degrees(altitude)
    if apply_refraction:
        altitude_deg += refraction_bennett(altitude_deg)

    return altitude_deg, math.degrees(azimuth) % 360


def sun_track(start_cest: float = 19.0,
              end_cest: float = 21.25,
              step_seconds: int = 30,
              **kwargs) -> pd.DataFrame:
    """Tabulate the sun's path across the eclipse window.

    Returns a DataFrame with utc_hours, cest, altitude_deg, azimuth_deg.
    This table is the 'target' that horizon profiles get compared against.
    """
    rows = []
    t = (start_cest - UTC_OFFSET_HOURS) * 3600.0
    end = (end_cest - UTC_OFFSET_HOURS) * 3600.0

    while t <= end:
        hour_utc = t / 3600.0
        alt, az = sun_altaz(hour_utc, **kwargs)
        rows.append(SunPosition(
            utc_hours=hour_utc,
            cest=format_cest(hour_utc + UTC_OFFSET_HOURS),
            altitude_deg=alt,
            azimuth_deg=az,
        ))
        t += step_seconds

    return pd.DataFrame([vars(r) for r in rows])


def at_time(track: pd.DataFrame, cest_hours: float) -> pd.Series:
    """Nearest row in a track to a given local time, e.g. 20.2 for 20:12."""
    target_utc = cest_hours - UTC_OFFSET_HOURS
    return track.iloc[(track["utc_hours"] - target_utc).abs().argmin()]


def blocking_height(distance_m: np.ndarray | float,
                    altitude_deg: float) -> np.ndarray | float:
    """Height an object at a given distance needs, to block a given altitude.

    Pure geometry, ignoring curvature - a quick intuition tool rather than an
    analysis function. This is what shows that the far field barely matters:
    blocking an 8 deg sun from 5 km away takes a 700 m tower.
    """
    return np.tan(np.radians(altitude_deg)) * np.asarray(distance_m)
