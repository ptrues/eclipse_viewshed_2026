"""
Vector data from the Digitaal Vlaanderen WFS.

Service: https://geo.api.vlaanderen.be/GRB/wfs   (OGC WFS 2.0.0, free)

GRB is the Flemish large-scale reference file. The layer that matters here is
`GRB:GBG` — *gebouw aan de grond*, building footprints.

What GRB does NOT provide, verified against the live service:

- **No height.** GBG is 2D geometry only.
- **No construction date.** `OPNDATUM`, `BEGINDATUM` and `VERSDATUM` record when
  GRB surveyed or revised the feature, not when the building was built. A
  19th-century townhouse mapped in 2013 carries `OPNDATUM 2013-01-08`.

So footprints answer "is this obstruction a building or a tree", and nothing
else. Heights for post-lidar structures have to come from elsewhere.

See references/data_sources.md for the full provenance record.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests

from .aoi import Bounds

GRB_WFS = "https://geo.api.vlaanderen.be/GRB/wfs"
WFS_VERSION = "2.0.0"

LAYER_BUILDINGS = "GRB:GBG"

# Define the building and road feature-service endpoints.
WEGENREGISTER_WFS = "https://geo.api.vlaanderen.be/Wegenregister/wfs"
LAYER_ROADS = "Wegenregister:Wegsegment"

CRS_L72 = "EPSG:31370"
DEFAULT_PAGE = 1000
DEFAULT_TIMEOUT = 120


def feature_url(layer: str, bounds: Bounds, start: int = 0,
                count: int = DEFAULT_PAGE, url: str = GRB_WFS) -> str:
    """A single paged GetFeature request, as GeoJSON in Lambert 72."""
    bbox = f"{bounds.xmin:.0f},{bounds.ymin:.0f},{bounds.xmax:.0f},{bounds.ymax:.0f},{CRS_L72}"
    return (
        f"{url}?service=WFS&version={WFS_VERSION}&request=GetFeature"
        f"&typeName={layer.replace(':', '%3A')}"
        f"&outputFormat=application%2Fjson"
        f"&srsName={CRS_L72}"
        f"&bbox={bbox}"
        f"&count={count}&startIndex={start}"
    )


def cache_name(layer: str, bounds: Bounds, tag: str = "") -> str:
    """Cache filename encoding the area requested.

    A fixed filename would be reused after the area of interest moves — change
    the primary observer and you would silently analyse footprints from the old
    location. Encoding the bounds makes a different request a different file.
    """
    stem = layer.replace(":", "_")
    box = (f"x{bounds.xmin:.0f}_{bounds.xmax:.0f}"
           f"_y{bounds.ymin:.0f}_{bounds.ymax:.0f}")
    return f"{stem}_{box}{('_' + tag) if tag else ''}.geojson"


def fetch_features(layer: str, bounds: Bounds,
                   out_path: Path | None = None,
                   page_size: int = DEFAULT_PAGE,
                   max_pages: int = 200,
                   timeout: int = DEFAULT_TIMEOUT,
                   overwrite: bool = False,
                   progress: bool = True,
                   url: str = GRB_WFS) -> dict:
    """Fetch and optionally cache all features intersecting the bounds."""
    if out_path is not None:
        out_path = Path(out_path)
        if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
            if progress:
                print(f"  {out_path.name}: from cache")
            return json.loads(out_path.read_text(encoding="utf-8"))

    features: list[dict] = []
    for page in range(max_pages):
        start = page * page_size
        r = requests.get(feature_url(layer, bounds, start, page_size, url=url),
                         timeout=timeout)
        r.raise_for_status()

        # WFS reports errors as XML with HTTP 200, so check the payload.
        ctype = r.headers.get("Content-Type", "")
        if "json" not in ctype.lower():
            raise RuntimeError(
                f"{layer}: expected GeoJSON, got {ctype}\n"
                f"{r.text[:400]}")

        batch = r.json().get("features", [])
        features.extend(batch)
        if progress:
            print(f"  {layer}: {len(features)} features", end="\r")
        if len(batch) < page_size:
            break
    else:
        raise RuntimeError(
            f"{layer}: hit max_pages={max_pages}; widen page_size or narrow bounds")

    if progress:
        print()

    collection = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": f"urn:ogc:def:crs:{CRS_L72}"}},
        "features": features,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(collection), encoding="utf-8")
    return collection


def geometries(collection: dict) -> list[dict]:
    """Bare GeoJSON geometries, for rasterisation."""
    return [f["geometry"] for f in collection.get("features", [])
            if f.get("geometry")]
