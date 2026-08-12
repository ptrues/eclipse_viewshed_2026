"""
Horizon profiles: for each bearing, how high does the ground rise?

This is the analysis the rest of the project exists to support. For one
observer it casts a fan of rays across the azimuth wedge the eclipsed sun
crosses, samples the surface model along each, and records the maximum
elevation angle any obstruction reaches. Comparing that profile against the
sun's track gives the time the sun is lost.

Three corrections that matter at these very low solar altitudes
---------------------------------------------------------------
**Earth curvature and refraction.** The ground falls away from a flat sightline
by d²(1−k)/2R. At 5 km that is 1.7 m and at 10 km it is 6.8 m — comparable to
the height differences deciding the answer. k = 0.13 is the standard
coefficient of terrestrial refraction, which offsets about 13% of the drop.

**Self-occlusion.** An observer on a roof or a raised terrace stands on a DSM
cell as tall as they are. Sampling from the first pixel outward would report
that cell as an obstruction at ~90°. Rays therefore start at `near_m`.

**Class exclusion.** Water is flat and near sea level and cannot obstruct, but
lidar over water is noisy and an isolated spurious return would set a false
horizon. Where a classified raster is supplied, those pixels are dropped.

What this does NOT model
------------------------
The sun as a disc. The profile compares against the sun's centre; the disc is
0.53° across, so the last sliver survives roughly 1.7 minutes longer than the
centre-based answer. `visibility()` reports both.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds

from .solar import UTC_OFFSET_HOURS, WEDGE_AZ_MAX, WEDGE_AZ_MIN

R_EARTH_M = 6_371_000.0

# Standard coefficient of terrestrial refraction. Light bends toward the Earth,
# offsetting ~13% of the geometric drop.
K_REFRACTION = 0.13

# Angular radius of the solar disc, degrees. Varies ~1.7% over the year; the
# variation is far below our resolution.
SUN_DISC_RADIUS_DEG = 0.265

# Ray sampling. Dense close in, where a metre of position matters, coarser far
# out, where it does not.
NEAR_MAX_M, NEAR_STEP_M = 2000.0, 2.0
FAR_STEP_M = 10.0

# Skip the observer's immediate surroundings, so a rooftop or terrace does not
# occlude itself.
DEFAULT_NEAR_M = 20.0

DEFAULT_AZ_STEP = 0.1


@dataclass
class HorizonProfile:
    """Maximum obstruction angle per bearing, and what caused it."""
    azimuth_deg: np.ndarray
    horizon_deg: np.ndarray
    distance_m: np.ndarray        # range to the controlling obstruction
    height_taw: np.ndarray        # its elevation
    class_code: np.ndarray        # its class, 0 if unclassified
    z_eye_taw: float
    name: str = ""
    #: Fraction of the analysed area the classification actually covered. Below
    #: 1.0 means the classes raster is older than the DSM mosaic — re-run
    #: notebook 03. Uncovered pixels are treated as unclassified, so nothing is
    #: wrongly excluded, but water goes unmasked there.
    class_coverage: float = 1.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "azimuth_deg": self.azimuth_deg,
            "horizon_deg": self.horizon_deg,
            "distance_m": self.distance_m,
            "height_taw": self.height_taw,
            "class_code": self.class_code,
        })

    def at_azimuth(self, az: float) -> float:
        """Horizon angle at one bearing, linearly interpolated."""
        return float(np.interp(az, self.azimuth_deg, self.horizon_deg))


def curvature_drop(distance_m, k: float = K_REFRACTION):
    """How far the surface falls below a straight sightline, in metres."""
    d = np.asarray(distance_m, dtype="float64")
    return d ** 2 * (1.0 - k) / (2.0 * R_EARTH_M)


def sample_distances(ray_m: float, near_m: float = DEFAULT_NEAR_M) -> np.ndarray:
    """Distances at which to sample each ray."""
    near_end = min(NEAR_MAX_M, ray_m)
    near = np.arange(near_m, near_end, NEAR_STEP_M)
    if ray_m <= NEAR_MAX_M:
        return near
    far = np.arange(near_end, ray_m, FAR_STEP_M)
    return np.concatenate([near, far])


class OutsideCoverage(ValueError):
    """An observer lies beyond the acquired rasters."""


def _require_inside(src, x: float, y: float, path) -> None:
    """Raise a legible error if an observer falls outside the raster.

    Without this, rasterio reports `WindowError: Intersection is empty`, which
    says nothing about the actual problem: the observer's tiles were never
    downloaded. Notebook 02 must fetch the union of every observer's fan, not
    just the primary's.
    """
    b = src.bounds
    if not (b.left <= x <= b.right and b.bottom <= y <= b.top):
        raise OutsideCoverage(
            f"observer at ({x:.0f}, {y:.0f}) is outside {Path(path).name} "
            f"[X {b.left:.0f}-{b.right:.0f}, Y {b.bottom:.0f}-{b.top:.0f}]. "
            "Re-run notebook 02 so this observer's tiles are fetched.")


def _read_aligned(path: Path, ref_transform, ref_shape, ref_res):
    """Read a raster onto another raster's window, by geography not by index.

    Reusing the DSM's window indices against a different raster only works if
    the two share an origin as well as a resolution. They frequently do not:
    re-running notebook 02 with an extra observer grows the DSM mosaic, leaving
    the classified raster from notebook 03 covering a smaller area on the same
    1 m grid.

    So the window is derived from world coordinates and read boundlessly.
    Pixels the source does not cover come back as 0 (unclassified), which is
    the safe default — nothing gets wrongly excluded from the ray-cast.

    Returns (array, covered_fraction).
    """
    height, width = ref_shape
    west, north = ref_transform * (0, 0)
    east, south = ref_transform * (width, height)

    with rasterio.open(path) as src:
        if not np.allclose(src.res, ref_res, rtol=0, atol=1e-6):
            raise ValueError(
                f"{Path(path).name} has resolution {src.res}, but the DSM is "
                f"{ref_res}. They must share a grid; rebuild it from the DSM.")

        win = window_from_bounds(west, south, east, north,
                                 transform=src.transform)
        # Fix the size to the reference so the arrays always align, and round
        # the offset to whole pixels.
        win = Window(round(win.col_off), round(win.row_off), width, height)
        arr = src.read(1, window=win, boundless=True, fill_value=0)

        overlap_w = max(0.0, min(east, src.bounds.right) - max(west, src.bounds.left))
        overlap_h = max(0.0, min(north, src.bounds.top) - max(south, src.bounds.bottom))
        area = (east - west) * (north - south)
        covered = (overlap_w * overlap_h / area) if area > 0 else 0.0

    return arr, covered


def resolve_eye_z(x: float, y: float,
                  dsm_path: Path, dtm_path: Path,
                  z_mode: str = "ground",
                  eye_height_m: float = 1.6,
                  abs_eye_z_taw: float | None = None,
                  sample_m: int = 5) -> tuple[float, float]:
    """Absolute eye elevation in m TAW. Returns (z_eye, base_elevation).

    `z_mode` selects the datum:
      ground   - DTM at the point, plus eye_height_m. Standing on bare earth.
      surface  - DSM at the point, plus eye_height_m. Standing on whatever the
                 lidar saw, which for a roof is the roof.
      absolute - abs_eye_z_taw verbatim, for structures absent from the DSM.

    The base elevation is the median of a small window rather than a single
    pixel, so one noisy return cannot move the observer.
    """
    if z_mode == "absolute":
        if abs_eye_z_taw is None or not np.isfinite(abs_eye_z_taw):
            raise ValueError("z_mode='absolute' requires abs_eye_z_taw")
        return float(abs_eye_z_taw), float(abs_eye_z_taw)

    if z_mode not in ("ground", "surface"):
        raise ValueError(f"unknown z_mode {z_mode!r}; "
                         "expected 'ground', 'surface' or 'absolute'")

    path = dtm_path if z_mode == "ground" else dsm_path
    half = sample_m / 2.0
    with rasterio.open(path) as src:
        _require_inside(src, x, y, path)
        win = window_from_bounds(x - half, y - half, x + half, y + half,
                                 transform=src.transform)
        win = win.round_offsets().round_lengths()
        win = win.intersection(Window(0, 0, src.width, src.height))
        patch = src.read(1, window=win, masked=True)

    if patch.count() == 0:
        raise ValueError(f"no valid elevation at ({x:.0f}, {y:.0f}) in {path.name}")

    base = float(np.ma.median(patch))
    return base + float(eye_height_m), base


def compute_horizon(dsm_path: Path, x0: float, y0: float, z_eye: float,
                    classes_path: Path | None = None,
                    exclude_classes: tuple[int, ...] = (),
                    ray_m: float = 10_000.0,
                    near_m: float = DEFAULT_NEAR_M,
                    az_min: float = WEDGE_AZ_MIN,
                    az_max: float = WEDGE_AZ_MAX,
                    az_step: float = DEFAULT_AZ_STEP,
                    name: str = "") -> HorizonProfile:
    """Cast a fan of rays and record the highest obstruction on each.

    The surface is read once into memory for the fan's bounding box and then
    indexed by nearest neighbour. Sampling point by point through rasterio
    would issue hundreds of thousands of tiny reads.
    """
    azimuths = np.arange(az_min, az_max + az_step / 2, az_step)
    dists = sample_distances(ray_m, near_m)
    # The drop is computed inside the loop, against the distances that survive
    # clipping and masking, so it stays aligned with the sampled heights.

    # Bounding box of the fan, clipped to the raster.
    ang = np.radians(np.concatenate([azimuths, [az_min, az_max]]))
    fx = x0 + np.sin(ang) * ray_m
    fy = y0 + np.cos(ang) * ray_m
    xmin, xmax = min(fx.min(), x0), max(fx.max(), x0)
    ymin, ymax = min(fy.min(), y0), max(fy.max(), y0)

    with rasterio.open(dsm_path) as src:
        _require_inside(src, x0, y0, dsm_path)
        b = src.bounds
        win = window_from_bounds(max(xmin, b.left), max(ymin, b.bottom),
                                 min(xmax, b.right), min(ymax, b.top),
                                 transform=src.transform)
        win = win.round_offsets().round_lengths()
        win = win.intersection(Window(0, 0, src.width, src.height))
        if win.width <= 0 or win.height <= 0:
            raise ValueError("observer's fan falls outside the raster")
        surf = src.read(1, window=win, masked=True)
        win_transform = src.window_transform(win)
        dsm_res = src.res

    classes, class_coverage = None, 1.0
    if classes_path is not None:
        classes, class_coverage = _read_aligned(
            classes_path, win_transform, surf.shape, dsm_res)

    inv = ~win_transform
    rows, cols = surf.shape
    z = np.ma.filled(surf, np.nan).astype("float64")
    invalid = np.ma.getmaskarray(surf)
    if classes is not None and exclude_classes:
        invalid = invalid | np.isin(classes, exclude_classes)

    n = azimuths.size
    horizon = np.full(n, -90.0)
    at_dist = np.full(n, np.nan)
    at_height = np.full(n, np.nan)
    at_class = np.zeros(n, dtype="uint8")

    for i, az in enumerate(azimuths):
        rad = np.radians(az)
        xs = x0 + np.sin(rad) * dists
        ys = y0 + np.cos(rad) * dists

        c, r = inv * (xs, ys)
        ci = np.rint(np.asarray(c)).astype(np.int64)
        ri = np.rint(np.asarray(r)).astype(np.int64)

        inside = (ci >= 0) & (ci < cols) & (ri >= 0) & (ri < rows)
        if not inside.any():
            continue

        ri, ci = ri[inside], ci[inside]
        d = dists[inside]
        zz = z[ri, ci]
        ok = ~invalid[ri, ci] & np.isfinite(zz)
        if not ok.any():
            continue

        zz, d = zz[ok], d[ok]
        angles = np.degrees(np.arctan2(zz - z_eye - curvature_drop(d), d))

        j = int(np.argmax(angles))
        horizon[i] = angles[j]
        at_dist[i] = d[j]
        at_height[i] = zz[j]
        if classes is not None:
            at_class[i] = classes[ri[ok][j], ci[ok][j]]

    return HorizonProfile(azimuths, horizon, at_dist, at_height, at_class,
                          float(z_eye), name, float(class_coverage))


def visibility(profile: HorizonProfile, track: pd.DataFrame,
               max_eclipse_cest: float = 20.2,
               disc_radius_deg: float = SUN_DISC_RADIUS_DEG,
               eclipse_window_cest: tuple[float, float] | None = None) -> dict:
    """When is the sun above this horizon?

    Reported twice: for the sun's centre, and for the upper limb, which stays
    visible about 1.7 minutes longer because the disc is 0.53° across.

    A NaN horizon (bearing outside the profile) counts as blocked, which is the
    conservative reading.

    `eclipse_window_cest` is (first contact, last contact) in local hours, from
    `eclipse.contacts()`. Supplying it adds the only figure a reader actually
    wants: how many minutes of the eclipse are visible from this spot. That is
    bounded at the start by first contact - before which there is nothing to
    see - and at the end by the sun going behind the measured skyline, which
    here always happens before last contact. The sun's upper limb is used, so
    the clock runs until the last sliver disappears.

    Without the window the eclipse fields are omitted, and the remaining
    figures describe the sun's whole passage across the analysed wedge, which
    starts when its bearing enters the wedge rather than at any solar event.
    """
    az = track["azimuth_deg"].to_numpy()
    alt = track["altitude_deg"].to_numpy()
    hor = np.interp(az, profile.azimuth_deg, profile.horizon_deg,
                    left=np.nan, right=np.nan)

    out: dict = {"name": profile.name, "z_eye_taw": round(profile.z_eye_taw, 2)}

    idx_max = int(np.argmin(np.abs(track["utc_hours"].to_numpy()
                                   - (max_eclipse_cest - 2.0))))
    out["sun_alt_at_max"] = round(float(alt[idx_max]), 2)
    out["horizon_at_max"] = (round(float(hor[idx_max]), 2)
                             if np.isfinite(hor[idx_max]) else None)
    out["clearance_at_max"] = (round(float(alt[idx_max] - hor[idx_max]), 2)
                               if np.isfinite(hor[idx_max]) else None)
    out["visible_at_max"] = bool(np.isfinite(hor[idx_max])
                                 and alt[idx_max] > hor[idx_max])

    for label, margin in (("centre", 0.0), ("upper_limb", disc_radius_deg)):
        clear = np.isfinite(hor) & (alt + margin > hor)
        if not clear.any():
            out[f"first_visible_{label}"] = None
            out[f"last_visible_{label}"] = None
            out[f"minutes_visible_{label}"] = 0.0
            continue
        idx = np.flatnonzero(clear)
        # Contiguous run containing the earliest visible sample.
        breaks = np.flatnonzero(np.diff(idx) > 1)
        end = idx[breaks[0]] if breaks.size else idx[-1]
        out[f"first_visible_{label}"] = track.iloc[idx[0]]["cest"]
        out[f"last_visible_{label}"] = track.iloc[end]["cest"]
        step_min = float(np.diff(track["utc_hours"].to_numpy()[:2])[0] * 60)
        out[f"minutes_visible_{label}"] = round(
            (end - idx[0] + 1) * step_min, 1)

    if eclipse_window_cest is not None:
        c1, c4 = eclipse_window_cest
        local = track["utc_hours"].to_numpy() + UTC_OFFSET_HOURS
        in_window = (local >= c1) & (local <= c4)
        # Upper limb: any sliver above the skyline still counts as visible.
        clear = in_window & np.isfinite(hor) & (alt + disc_radius_deg > hor)
        if not clear.any():
            out["eclipse_first_cest"] = None
            out["eclipse_last_cest"] = None
            out["eclipse_minutes"] = 0.0
        else:
            idx = np.flatnonzero(clear)
            breaks = np.flatnonzero(np.diff(idx) > 1)
            end = idx[breaks[0]] if breaks.size else idx[-1]
            out["eclipse_first_cest"] = track.iloc[idx[0]]["cest"]
            out["eclipse_last_cest"] = track.iloc[end]["cest"]
            out["eclipse_minutes"] = round(
                (local[end] - local[idx[0]]) * 60, 1)
            # True when the profile, not the skyline, ended the run.
            out["eclipse_clipped_by_wedge"] = bool(
                end + 1 < len(local)
                and in_window[end + 1]
                and not np.isfinite(hor[end + 1]))

    return out


def controlling_classes(profile: HorizonProfile, labels: dict) -> pd.DataFrame:
    """Share of bearings whose horizon is set by each class."""
    codes, counts = np.unique(profile.class_code, return_counts=True)
    total = counts.sum()
    rows = []
    for code, n in zip(codes.tolist(), counts.tolist()):
        m = profile.class_code == code
        rows.append({
            "class": labels.get(code, f"code {code}"),
            "bearings": n,
            "share": f"{100 * n / total:.1f}%",
            "median_horizon_deg": round(float(np.median(profile.horizon_deg[m])), 2),
            "max_horizon_deg": round(float(np.max(profile.horizon_deg[m])), 2),
            "median_distance_m": (round(float(np.nanmedian(profile.distance_m[m])))
                                  if np.isfinite(profile.distance_m[m]).any() else None),
        })
    return pd.DataFrame(rows).sort_values("bearings", ascending=False)

def ray_profile(dsm_path: Path, x0: float, y0: float, z_eye: float,
                azimuth_deg: float,
                ray_m: float = 10_000.0,
                near_m: float = DEFAULT_NEAR_M,
                classes_path: Path | None = None) -> pd.DataFrame:
    """Ground elevation along one bearing — the terrain cross-section.

    `compute_horizon` reduces each ray to a single number, which is what the
    analysis needs but gives a reader nothing to judge. A cross-section is
    directly legible: the river is a flat low stretch, the far bank is a step,
    trees are a ragged band on top of it. Someone who knows the place can look
    at this and say whether the model matches the ground.

    Returns distance, surface elevation, the curvature-corrected apparent
    angle, and the class code where available.
    """
    dists = sample_distances(ray_m, near_m)
    rad = np.radians(azimuth_deg)
    xs = x0 + np.sin(rad) * dists
    ys = y0 + np.cos(rad) * dists

    with rasterio.open(dsm_path) as src:
        _require_inside(src, x0, y0, dsm_path)
        inv = ~src.transform
        c, r = inv * (xs, ys)
        ci = np.rint(np.asarray(c)).astype(np.int64)
        ri = np.rint(np.asarray(r)).astype(np.int64)
        inside = (ci >= 0) & (ci < src.width) & (ri >= 0) & (ri < src.height)

        z = np.full(dists.shape, np.nan)
        if inside.any():
            # One windowed read spanning the ray, then index into it.
            r0, r1 = int(ri[inside].min()), int(ri[inside].max()) + 1
            c0, c1 = int(ci[inside].min()), int(ci[inside].max()) + 1
            win = Window(c0, r0, c1 - c0, r1 - r0)
            block = src.read(1, window=win, masked=True)
            z[inside] = np.ma.filled(
                block[ri[inside] - r0, ci[inside] - c0], np.nan)

    cls = np.zeros(dists.shape, dtype="uint8")
    if classes_path is not None:
        with rasterio.open(classes_path) as src:
            inv = ~src.transform
            c, r = inv * (xs, ys)
            ci = np.rint(np.asarray(c)).astype(np.int64)
            ri = np.rint(np.asarray(r)).astype(np.int64)
            ok = (ci >= 0) & (ci < src.width) & (ri >= 0) & (ri < src.height)
            if ok.any():
                r0, r1 = int(ri[ok].min()), int(ri[ok].max()) + 1
                c0, c1 = int(ci[ok].min()), int(ci[ok].max()) + 1
                block = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
                cls[ok] = block[ri[ok] - r0, ci[ok] - c0]

    drop = curvature_drop(dists)
    return pd.DataFrame({
        "distance_m": dists,
        "elevation_taw": z,
        # Elevation adjusted for curvature and refraction, i.e. what the eye
        # effectively sees. Plotting this rather than raw elevation makes the
        # sightline a straight line.
        "apparent_taw": z - drop,
        "angle_deg": np.degrees(np.arctan2(z - z_eye - drop, dists)),
        "class_code": cls,
    })


def controlling_feature(profile: HorizonProfile, azimuth_deg: float) -> dict:
    """What sets the horizon at one bearing — the evidence behind a number.

    A clearance figure on its own is unfalsifiable. Naming the object that
    produced it, with its range, height and class, lets a reader check it
    against the place.
    """
    i = int(np.argmin(np.abs(profile.azimuth_deg - azimuth_deg)))
    return {
        "azimuth_deg": round(float(profile.azimuth_deg[i]), 2),
        "horizon_deg": round(float(profile.horizon_deg[i]), 2),
        "distance_m": (round(float(profile.distance_m[i]))
                       if np.isfinite(profile.distance_m[i]) else None),
        "height_taw": (round(float(profile.height_taw[i]), 1)
                       if np.isfinite(profile.height_taw[i]) else None),
        "height_above_eye_m": (round(float(profile.height_taw[i] - profile.z_eye_taw), 1)
                               if np.isfinite(profile.height_taw[i]) else None),
        "class_code": int(profile.class_code[i]),
    }

def describe_surroundings(x: float, y: float, dsm_path: Path, dtm_path: Path,
                          classes_path: Path | None = None,
                          radius_m: float = 300.0,
                          obstruction_classes: tuple[int, ...] = (3, 4, 5),
                          water_class: int = 2) -> dict:
    """What is immediately around an observer — a placement check.

    A clearance figure cannot tell you whether the coordinate is where you meant
    it to be, and a misplaced point produces a confident wrong answer rather
    than an error. This answers the question directly: a point on a riverbank
    should sit within metres of water with nothing tall nearby, whereas a point
    dropped into a city block sits far from water with a building next door.

    Returns distances to the nearest water and the nearest obstruction, the
    tallest thing close by, and the ground elevation.
    """
    half = radius_m
    with rasterio.open(dsm_path) as src:
        _require_inside(src, x, y, dsm_path)
        b = src.bounds
        win = window_from_bounds(max(x - half, b.left), max(y - half, b.bottom),
                                 min(x + half, b.right), min(y + half, b.top),
                                 transform=src.transform)
        win = win.round_offsets().round_lengths()
        win = win.intersection(Window(0, 0, src.width, src.height))
        dsm = src.read(1, window=win, masked=True)
        win_transform = src.window_transform(win)
        res = src.res

    with rasterio.open(dtm_path) as src:
        dtm, _ = _read_aligned(dtm_path, win_transform, dsm.shape, res)

    height = np.ma.filled(dsm, np.nan) - dtm

    # Pixel-centre coordinates for the window, then range from the observer.
    t = win_transform
    cols = np.arange(dsm.shape[1])
    rows = np.arange(dsm.shape[0])
    xs = (t.c + (cols + 0.5) * t.a)[np.newaxis, :]
    ys = (t.f + (rows + 0.5) * t.e)[:, np.newaxis]
    dist = np.hypot(xs - x, ys - y)

    out: dict = {
        "ground_taw": round(float(np.nanmedian(dtm[dist <= 10])), 2),
        "tallest_within_50m": round(float(np.nanmax(
            np.where(dist <= 50, height, np.nan))), 1),
        "tallest_within_200m": round(float(np.nanmax(
            np.where(dist <= 200, height, np.nan))), 1),
    }

    if classes_path is not None:
        cls, _ = _read_aligned(classes_path, win_transform, dsm.shape, res)

        water = cls == water_class
        out["nearest_water_m"] = (round(float(dist[water].min()))
                                  if water.any() else None)
        out["water_within_100m_pct"] = round(
            100.0 * float(water[dist <= 100].mean()), 1)

        obstruction = np.isin(cls, obstruction_classes)
        out["nearest_obstruction_m"] = (round(float(dist[obstruction].min()))
                                        if obstruction.any() else None)
        out["class_at_point"] = int(cls[dist.argmin() // dist.shape[1],
                                        dist.argmin() % dist.shape[1]])

    return out
