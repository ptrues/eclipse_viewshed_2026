"""
Area-of-interest geometry and the tile grid.

Pure geometry — no network, no file I/O — so it is cheap to test.

The design goal is that adding a second observation point costs only the data
that observer needs and nobody else has already fetched. That is achieved by
tiling on a *fixed global grid* rather than on each observer's bounding box.

A tile is identified by the Lambert 72 coordinates of its lower-left corner,
snapped to a multiple of TILE_M. Tile (151000, 211000) means the same square of
ground no matter which observer requested it, so the on-disk cache is shared
across observers and across sessions.

If tiles were cut relative to each AOI, two overlapping AOIs would produce two
different, non-interchangeable sets of files covering the same ground.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pyproj import Transformer

from .solar import WEDGE_AZ_MAX, WEDGE_AZ_MIN

# Belgian Lambert 72. Projected, metres — required for ray casting.
CRS_ANALYSIS = "EPSG:31370"
CRS_INPUT = "EPSG:4326"

# Tile edge length in metres. Also the pixel count per tile at 1 m resolution,
# so 1000 => 1000x1000 px => ~4 MB per uncompressed float32 request.
# Raise if the WCS tolerates larger responses; lower if it refuses.
#
# Functions below take `tile_m=None` and resolve it at CALL time rather than
# declaring `tile_m=TILE_M` in the signature. Python evaluates default
# arguments once, when the function is defined, so a signature default would
# freeze the value at import and reassigning TILE_M afterwards would silently
# do nothing.
TILE_M = 1000

# How far to ray-cast. At 1 deg solar altitude an obstruction 10 km away needs
# ~175 m of height to matter, which the Waaslandhaven wind turbines meet.
# Beyond that, essentially nothing in Flanders qualifies.
DEFAULT_RAY_M = 10_000

_to_l72 = Transformer.from_crs(CRS_INPUT, CRS_ANALYSIS, always_xy=True)
_to_wgs = Transformer.from_crs(CRS_ANALYSIS, CRS_INPUT, always_xy=True)


@dataclass(frozen=True)
class Bounds:
    """Axis-aligned bounding box in Lambert 72 metres."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area_km2(self) -> float:
        return self.width * self.height / 1e6

    def union(self, other: "Bounds") -> "Bounds":
        return Bounds(min(self.xmin, other.xmin), min(self.ymin, other.ymin),
                      max(self.xmax, other.xmax), max(self.ymax, other.ymax))

    def to_wgs84(self) -> tuple[float, float, float, float]:
        """(lon_min, lat_min, lon_max, lat_max), for drawing on a web map."""
        lon0, lat0 = _to_wgs.transform(self.xmin, self.ymin)
        lon1, lat1 = _to_wgs.transform(self.xmax, self.ymax)
        return lon0, lat0, lon1, lat1

    def __repr__(self) -> str:
        return (f"Bounds(X {self.xmin:.0f}-{self.xmax:.0f}, "
                f"Y {self.ymin:.0f}-{self.ymax:.0f}, "
                f"{self.width/1000:.1f}x{self.height/1000:.1f} km)")


def to_lambert72(lat: float, lon: float) -> tuple[float, float]:
    """WGS84 lat/lon -> Lambert 72 X/Y in metres."""
    return _to_l72.transform(lon, lat)


def to_wgs84(x: float, y: float) -> tuple[float, float]:
    """Lambert 72 X/Y -> WGS84 (lon, lat), for pasting into a map.

    Note the argument order: this returns (lon, lat) to match pyproj's
    always_xy convention, whereas `to_lambert72` takes (lat, lon) because that
    is how coordinates are written in observers.csv.
    """
    return _to_wgs.transform(x, y)


def fan_bounds(x: float, y: float,
               ray_m: float | None = None,
               az_min: float = WEDGE_AZ_MIN,
               az_max: float = WEDGE_AZ_MAX,
               pad_m: float = 200.0) -> Bounds:
    """Bounding box of the wedge an observer at (x, y) can see the sun through.

    Only the sector between az_min and az_max matters, and the sun is in the
    west-northwest, so this box is strongly lopsided: it extends ~10 km west
    and only a few hundred metres east. Requesting a symmetric square around
    the observer would roughly quadruple the download for no benefit.

    The extremes of a circular sector lie either at the arc endpoints or at a
    cardinal direction the sector crosses, so both are checked.
    """
    ray_m = DEFAULT_RAY_M if ray_m is None else ray_m
    xs, ys = [0.0], [0.0]                      # observer at the origin

    # Arc endpoints.
    for az in (az_min, az_max):
        xs.append(math.sin(math.radians(az)) * ray_m)
        ys.append(math.cos(math.radians(az)) * ray_m)

    # Any cardinal direction the sector sweeps past is also an extremum.
    for card, (dx, dy) in {0: (0, 1), 90: (1, 0), 180: (0, -1), 270: (-1, 0)}.items():
        if az_min <= card <= az_max:
            xs.append(dx * ray_m)
            ys.append(dy * ray_m)

    return Bounds(x + min(xs) - pad_m, y + min(ys) - pad_m,
                  x + max(xs) + pad_m, y + max(ys) + pad_m)


# --------------------------------------------------------------- tile grid

def tile_key(x: float, y: float, tile_m: int | None = None) -> tuple[int, int]:
    """Lower-left corner of the tile containing (x, y), snapped to the grid."""
    tile_m = TILE_M if tile_m is None else tile_m
    return (int(math.floor(x / tile_m) * tile_m),
            int(math.floor(y / tile_m) * tile_m))


def tile_bounds(key: tuple[int, int], tile_m: int | None = None) -> Bounds:
    """Bounds of one tile, from its key."""
    tile_m = TILE_M if tile_m is None else tile_m
    return Bounds(key[0], key[1], key[0] + tile_m, key[1] + tile_m)


def tile_name(key: tuple[int, int], tile_m: int | None = None) -> str:
    """Stable filename stem. Encodes the grid size so tiles cut at different
    resolutions cannot be silently mixed in one cache directory."""
    tile_m = TILE_M if tile_m is None else tile_m
    return f"x{key[0]}_y{key[1]}_t{tile_m}"


def tiles_covering(bounds: Bounds, tile_m: int | None = None) -> list[tuple[int, int]]:
    """Every grid tile intersecting the given bounds, ordered for readability."""
    tile_m = TILE_M if tile_m is None else tile_m
    x0, y0 = tile_key(bounds.xmin, bounds.ymin, tile_m)
    x1, y1 = tile_key(bounds.xmax, bounds.ymax, tile_m)
    return [(x, y)
            for y in range(y0, y1 + tile_m, tile_m)
            for x in range(x0, x1 + tile_m, tile_m)]


def tiles_for_observer(lat: float, lon: float,
                       ray_m: float | None = None,
                       tile_m: int | None = None) -> list[tuple[int, int]]:
    """Tiles needed to analyse one observation point, from WGS84 coordinates."""
    x, y = to_lambert72(lat, lon)
    return tiles_covering(fan_bounds(x, y, ray_m), tile_m)


#: Padding around the observers for the site map, in metres. Wider to the west
#: because that is where the sightline wedges point.
MAP_PAD_W, MAP_PAD_E, MAP_PAD_N, MAP_PAD_S = 1500, 500, 500, 500


#: Extra metres fetched beyond the map frame. Acquisition boxes are bigger
#: than the view on purpose: a layer that stops exactly at the frame edge shows
#: its own bounding box as a straight line across the picture, and a footprint
#: box that ends mid-frame turns the ground beyond it into class 5, "tall,
#: unattributed", rather than into buildings.
DATA_MARGIN_M = 2000


def map_frame(points, pad_w=MAP_PAD_W, pad_e=MAP_PAD_E,
              pad_n=MAP_PAD_N, pad_s=MAP_PAD_S, margin=0.0) -> "Bounds":
    """Bounds of the site map, from (lat, lon) observer positions.

    Defined here rather than in the notebook that draws the map, because the
    notebook that *acquires* data has to know about it too. The elevation tiles
    were originally chosen to cover the sightline fans alone, which point
    west-north-west; the map frame extends south-east behind the observers,
    where no ray goes. The result was a map with a nodata corner - the raster's
    extent was a function of the analysis rather than of the picture.
    """
    xs, ys = [], []
    for lat, lon in points:
        x, y = to_lambert72(lat, lon)
        xs.append(x)
        ys.append(y)
    return Bounds(min(xs) - pad_w - margin, min(ys) - pad_s - margin,
                  max(xs) + pad_e + margin, max(ys) + pad_n + margin)


def tiles_for_map(points, margin=DATA_MARGIN_M, **kwargs) -> list[tuple[int, int]]:
    """Tiles covering the site map frame, plus the acquisition margin."""
    return tiles_covering(map_frame(points, margin=margin, **kwargs))


def tiles_for_observers(points: list[tuple[float, float]],
                        ray_m: float | None = None,
                        tile_m: int | None = None) -> list[tuple[int, int]]:
    """Deduplicated union of tiles for several observers.

    Antwerp viewpoints all look west-northwest, so their fans overlap heavily
    and the union is far smaller than the sum. This is what makes adding a
    second viewpoint cheap.
    """
    seen: set[tuple[int, int]] = set()
    for lat, lon in points:
        seen.update(tiles_for_observer(lat, lon, ray_m, tile_m))
    return sorted(seen, key=lambda k: (k[1], k[0]))


def bearing_distance(x0: float, y0: float, xs, ys):
    """Azimuth and horizontal distance from an observer to grid points.

    Azimuth is degrees clockwise from true north, matching the convention used
    for the sun's position, so the two are directly comparable. Note the
    argument order in arctan2: `atan2(dx, dy)` gives a bearing from north,
    whereas the more familiar `atan2(dy, dx)` gives an angle from east.

    `xs` and `ys` broadcast, so passing a (1, W) row and an (H, 1) column
    yields (H, W) results without building meshgrids.
    """
    dx = np.asarray(xs) - x0
    dy = np.asarray(ys) - y0
    azimuth = np.degrees(np.arctan2(dx, dy)) % 360.0
    distance = np.hypot(dx, dy)
    return azimuth, distance


def in_wedge(azimuth, distance, max_distance: float,
             az_min: float = WEDGE_AZ_MIN, az_max: float = WEDGE_AZ_MAX,
             min_distance: float = 0.0):
    """Boolean mask of points inside the sightline sector.

    `min_distance` excludes the observer's immediate surroundings, which is how
    a rooftop observer avoids being occluded by the roof they are standing on.
    """
    return ((azimuth >= az_min) & (azimuth <= az_max)
            & (distance <= max_distance) & (distance >= min_distance))
