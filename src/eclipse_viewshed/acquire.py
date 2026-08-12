"""
Acquisition of DHMV II elevation rasters from the Digitaal Vlaanderen WCS.

Service: https://geo.api.vlaanderen.be/dhmv/wcs   (OGC WCS 2.0.1, free)

This is a *download* service, not a view service — GetCoverage returns real
elevation values as GeoTIFF, unlike the WMS layers of the same name which
return rendered pixels.

Design notes
------------
Tiles are cached on disk and never re-fetched. Combined with the fixed global
tile grid in `aoi.py`, this means analysing a second observation point only
downloads the tiles no previous observer needed.

Every fetch is recorded in a JSON manifest alongside the tiles: request URL,
bounds, timestamp, byte count and SHA-256. The manifest is the provenance
record — it is what lets someone else verify they got the same bytes, and it is
small enough to commit to version control even though the rasters are not.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

from . import aoi
from .aoi import Bounds, tile_bounds, tile_name

WCS_URL = "https://geo.api.vlaanderen.be/dhmv/wcs"
WCS_VERSION = "2.0.1"

COVERAGE_DSM = "DHMVII_DSM_1m"
COVERAGE_DTM = "DHMVII_DTM_1m"

# Namespaces used in the WCS 2.0.1 DescribeCoverage response.
NS = {
    "wcs": "http://www.opengis.net/wcs/2.0",
    "gml": "http://www.opengis.net/gml/3.2",
    "gmlcov": "http://www.opengis.net/gmlcov/1.0",
    "swe": "http://www.opengis.net/swe/2.0",
}

DEFAULT_TIMEOUT = 180


@dataclass
class CoverageInfo:
    """What the service says about a coverage. Read, never assumed."""
    coverage_id: str
    crs: str
    axis_labels: list[str]
    envelope: Bounds
    pixel_size: tuple[float, float]
    nodata: float | None
    native_format: str

    def __repr__(self) -> str:
        return (f"CoverageInfo({self.coverage_id}, {self.crs}, "
                f"axes={self.axis_labels}, px={self.pixel_size}, "
                f"nodata={self.nodata})")


def describe_coverage(coverage_id: str,
                      url: str = WCS_URL,
                      timeout: int = 60) -> CoverageInfo:
    """Parse DescribeCoverage into a structured record.

    Reading the grid definition from the service rather than hardcoding it
    means the notebook stays correct if Digitaal Vlaanderen re-grids the
    product, and it surfaces the axis label order, which the GetCoverage
    subset parameters depend on.
    """
    r = requests.get(url, params={
        "service": "WCS", "version": WCS_VERSION,
        "request": "DescribeCoverage", "coverageId": coverage_id,
    }, timeout=timeout)
    r.raise_for_status()
    root = ET.fromstring(r.content)

    env = root.find(".//gml:boundedBy/gml:Envelope", NS)
    lower = [float(v) for v in env.find("gml:lowerCorner", NS).text.split()]
    upper = [float(v) for v in env.find("gml:upperCorner", NS).text.split()]
    crs_uri = env.get("srsName", "")
    axis_labels = (env.get("axisLabels") or "x y").split()

    # Offset vectors give the pixel size and orientation. Typically (1,0) and
    # (0,-1) for a north-up 1 m grid.
    offsets = root.findall(".//gml:offsetVector", NS)
    vecs = [[float(v) for v in o.text.split()] for o in offsets]
    px = (abs(vecs[0][0]) if vecs else 1.0,
          abs(vecs[1][1]) if len(vecs) > 1 else 1.0)

    nil = root.find(".//swe:nilValue", NS)
    nodata = float(nil.text) if nil is not None and nil.text else None

    fmt = root.find(".//wcs:nativeFormat", NS)

    return CoverageInfo(
        coverage_id=coverage_id,
        crs="EPSG:" + crs_uri.rstrip("/").split("/")[-1] if crs_uri else "unknown",
        axis_labels=axis_labels,
        envelope=Bounds(lower[0], lower[1], upper[0], upper[1]),
        pixel_size=px,
        nodata=nodata,
        native_format=fmt.text if fmt is not None else "image/tiff",
    )


def coverage_url(coverage_id: str, bounds: Bounds,
                 axis_labels: tuple[str, str] = ("x", "y"),
                 url: str = WCS_URL) -> str:
    """Build a GetCoverage request URL for one bounding box.

    WCS 2.0.1 subsets by named axis, and the names come from DescribeCoverage
    (here 'x' and 'y'). Subset bounds are in the coverage's native CRS, so no
    subsettingCrs parameter is needed.
    """
    ax, ay = axis_labels
    return (
        f"{url}?service=WCS&version={WCS_VERSION}&request=GetCoverage"
        f"&coverageId={coverage_id}"
        f"&subset={ax}({bounds.xmin:.0f},{bounds.xmax:.0f})"
        f"&subset={ay}({bounds.ymin:.0f},{bounds.ymax:.0f})"
        f"&format=image/tiff"
    )


@dataclass
class FetchReport:
    """Outcome of an ensure_tiles run."""
    requested: int = 0
    downloaded: int = 0
    cached: int = 0
    failed: list[tuple[tuple[int, int], str]] = field(default_factory=list)
    paths: list[Path] = field(default_factory=list)
    bytes_downloaded: int = 0

    def __repr__(self) -> str:
        s = (f"FetchReport({self.requested} requested, {self.downloaded} downloaded, "
             f"{self.cached} already cached, {len(self.failed)} failed")
        if self.bytes_downloaded:
            s += f", {self.bytes_downloaded/1e6:.1f} MB"
        return s + ")"


#: GeoTIFF byte-order markers — 'II' little-endian, 'MM' big-endian.
TIFF_MAGIC = (b"II", b"MM")


def extract_geotiff(content: bytes, content_type: str | None) -> bytes:
    """Pull the GeoTIFF out of a GetCoverage response.

    WCS 2.0.1 does not necessarily return a bare raster. This service replies
    with `multipart/related`: the first part is a GML `RectifiedGridCoverage`
    describing the grid, the second is the GeoTIFF itself. Writing the whole
    response to a .tif produces a file GDAL cannot open.

    Handles three cases:
      - bare GeoTIFF, returned as-is
      - multipart, in which case the first part with TIFF magic bytes is used
      - anything else, which is treated as a server error and raised with the
        response text, since WCS reports failures as XML with HTTP 200

    Servers vary in whether they wrap the payload, so both forms are accepted
    rather than depending on a `mediaType` parameter the service may ignore.
    """
    if content[:2] in TIFF_MAGIC:
        return content

    ctype = (content_type or "").lower()
    if "multipart" in ctype:
        import email

        msg = email.message_from_bytes(
            b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + content
        )
        for part in msg.walk():
            payload = part.get_payload(decode=True)
            if payload and payload[:2] in TIFF_MAGIC:
                return payload
        raise RuntimeError(
            "multipart response contained no GeoTIFF part; "
            f"parts were {[p.get_content_type() for p in msg.walk()]}"
        )

    snippet = content[:400].decode("utf-8", errors="replace")
    raise RuntimeError(f"expected a GeoTIFF, got {content_type}\n{snippet}")


def fetch_tile(coverage_id: str, key: tuple[int, int], outdir: Path,
               axis_labels: tuple[str, str] = ("x", "y"),
               tile_m: int | None = None,
               timeout: int = DEFAULT_TIMEOUT,
               overwrite: bool = False) -> tuple[Path, bool]:
    """Fetch one tile. Returns (path, downloaded) — downloaded=False if cached.

    The response is unwrapped by `extract_geotiff`, so what lands on disk is
    always a plain GeoTIFF regardless of whether the server used multipart.
    """
    tile_m = aoi.TILE_M if tile_m is None else tile_m
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{coverage_id}_{tile_name(key, tile_m)}.tif"

    if path.exists() and path.stat().st_size > 0 and not overwrite:
        return path, False

    url = coverage_url(coverage_id, tile_bounds(key, tile_m), axis_labels)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()

    try:
        tiff = extract_geotiff(r.content, r.headers.get("Content-Type"))
    except RuntimeError as exc:
        raise RuntimeError(f"tile {key}: {exc}") from None

    path.write_bytes(tiff)
    return path, True


def ensure_tiles(coverage_id: str, keys: list[tuple[int, int]], outdir: Path,
                 axis_labels: tuple[str, str] = ("x", "y"),
                 tile_m: int | None = None,
                 pause_s: float = 0.1,
                 progress: bool = True) -> FetchReport:
    """Fetch every tile not already on disk, and update the manifest.

    Failures are collected rather than raised, so one bad tile does not lose
    the rest of a long run. Inspect `report.failed` and re-run — successful
    tiles are cached and skipped.
    """
    tile_m = aoi.TILE_M if tile_m is None else tile_m
    report = FetchReport(requested=len(keys))
    manifest_path = outdir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for i, key in enumerate(keys, 1):
        try:
            path, downloaded = fetch_tile(coverage_id, key, outdir,
                                          axis_labels, tile_m)
            report.paths.append(path)

            if downloaded:
                report.downloaded += 1
                data = path.read_bytes()
                report.bytes_downloaded += len(data)
                b = tile_bounds(key, tile_m)
                manifest[path.name] = {
                    "coverage": coverage_id,
                    "url": coverage_url(coverage_id, b, axis_labels),
                    "bounds_l72": [b.xmin, b.ymin, b.xmax, b.ymax],
                    "tile_m": tile_m,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                   time.gmtime()),
                }
                time.sleep(pause_s)          # be polite to a free public service
            else:
                report.cached += 1

            if progress and (i % 10 == 0 or i == len(keys)):
                print(f"  {i}/{len(keys)}  "
                      f"{report.downloaded} new, {report.cached} cached",
                      end="\r")
        except Exception as exc:
            report.failed.append((key, str(exc)[:200]))

    if progress:
        print()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return report


def build_vrt(paths: list[Path], out_path: Path) -> Path:
    """Mosaic tiles into a GDAL virtual raster.

    A VRT is an XML index, not a copy — it costs kilobytes and makes the whole
    mosaic addressable as a single raster. Downstream code can window-read it
    exactly as if it were one large GeoTIFF, and adding tiles later means
    rebuilding a small XML file rather than rewriting gigabytes.
    """
    from osgeo import gdal

    gdal.UseExceptions()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vrt = gdal.BuildVRT(str(out_path), [str(p) for p in sorted(paths)])
    if vrt is None:
        raise RuntimeError(f"BuildVRT failed for {len(paths)} tiles")
    vrt.FlushCache()
    del vrt
    return out_path
