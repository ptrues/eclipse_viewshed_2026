# Data sources

This project uses the following public datasets. Acquisition code and exact spatial requests are recorded in notebooks 02, 03 and 06.

## DHMV II elevation models

The obstruction surface uses the one-metre DHMV II digital surface model (DSM) and digital terrain model (DTM), acquired through Digitaal Vlaanderen's OGC WCS service:

`https://geo.api.vlaanderen.be/dhmv/wcs`

| Product | Coverage ID | Purpose |
|---|---|---|
| Surface model | `DHMVII_DSM_1m` | Elevation of terrain, buildings and vegetation |
| Terrain model | `DHMVII_DTM_1m` | Bare-earth elevation used to calculate object height |

DHMV II was surveyed in 2013–2015. The rasters use EPSG:31370, one-metre cells and TAW vertical heights. The project downloads fixed tiles and records them in `data/raw/dhmv_tiles/manifest.json`.

The accompanying OpenLiDAR manual is stored as `EODaS_openlidar_Handleiding.pdf`. Digitaal Vlaanderen's open-data conditions are available at <https://overheid.vlaanderen.be/voorwaarden>.

## GRB building footprints

Building footprints come from the `GRB:GBG` layer in Digitaal Vlaanderen's OGC WFS service:

`https://geo.api.vlaanderen.be/GRB/wfs`

The footprints distinguish buildings from vegetation in the elevation surface. They are two-dimensional and provide neither building heights nor reliable construction dates.

## ESA WorldCover

Permanent water is identified with class 80 from the ESA WorldCover 2021 product. Antwerp falls in tile `N51E003`, acquired from the project's public cloud-optimised GeoTIFFs.

WorldCover is licensed under CC BY 4.0. Attribution: © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by the ESA WorldCover consortium.

## Wegenregister and GRB map layers

Notebook 06 obtains road centre lines from Digitaal Vlaanderen's Wegenregister WFS and water and building geometry from GRB. These layers provide context for the viewing-location map and do not affect the visibility calculations.

- Wegenregister WFS: `https://geo.api.vlaanderen.be/Wegenregister/wfs`
- GRB WFS: `https://geo.api.vlaanderen.be/GRB/wfs`

## Reference photograph

`500px-Lange_Wapper_statue.jpg` is a reduced copy of Mark Ahsmann's 2010 photograph “Lange Wapper statue, Antwerp,” published on Wikimedia Commons under CC BY-SA 3.0 (among other offered licences). The factsheet uses a cropped and darkened adaptation and carries the required attribution and share-alike notice.

Source: <https://commons.wikimedia.org/wiki/File:Lange_Wapper_statue.JPG>

## Analytical limitations

- The 2013–2015 elevation surface does not include later construction or vegetation growth.
- Current building footprints may contain structures that postdate the elevation survey; they classify the recorded surface but do not add missing heights.
- WorldCover's ten-metre cells are used only to identify permanent water.
- All elevation calculations use TAW heights; ellipsoidal heights are not interchangeable with them.
