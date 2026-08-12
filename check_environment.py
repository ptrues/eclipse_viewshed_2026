"""Check the project's Python and geospatial runtime."""

from __future__ import annotations

import importlib
import shutil
import subprocess

REQUIRED = (
    "numpy",
    "pandas",
    "rasterio",
    "pyproj",
    "matplotlib",
    "requests",
    "bs4",
    "pytest",
)


def report(label: str, passed: bool, detail: str = "") -> bool:
    """Print one check result and return its Boolean value."""
    suffix = f" — {detail}" if detail else ""
    print(f"{'PASS' if passed else 'FAIL':<6} {label}{suffix}")
    return passed


def check_imports() -> bool:
    """Import each required Python package."""
    results = []
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "installed")
            results.append(report(name, True, str(version)))
        except ImportError as error:
            results.append(report(name, False, str(error)))
    return all(results)


def check_gdal() -> bool:
    """Confirm that the GDAL command-line tools are available."""
    results = []
    for command in ("gdalinfo", "gdalbuildvrt", "gdalwarp", "gdal_translate"):
        path = shutil.which(command)
        results.append(report(command, path is not None, path or "not on PATH"))

    if shutil.which("gdalinfo"):
        version = subprocess.run(
            ["gdalinfo", "--version"],
            capture_output=True,
            check=False,
            text=True,
            timeout=20,
        ).stdout.strip()
        report("GDAL version", bool(version), version)
    return all(results)


def check_transform() -> bool:
    """Run a known WGS84-to-Lambert 72 coordinate transform."""
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:31370", always_xy=True)
    x, y = transformer.transform(4.388558346421922, 51.21193309444983)
    valid = abs(x - 151384) < 50 and abs(y - 211331) < 50
    return report("WGS84 to Lambert 72", valid, f"X={x:.0f}, Y={y:.0f}")


def check_raster() -> bool:
    """Write and read an in-memory Lambert 72 raster."""
    import numpy as np
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    data = np.arange(25, dtype="float32").reshape(5, 5)
    profile = {
        "driver": "GTiff",
        "height": 5,
        "width": 5,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:31370",
        "transform": from_origin(151800, 211800, 1, 1),
    }
    with MemoryFile() as memory:
        with memory.open(**profile) as destination:
            destination.write(data, 1)
        with memory.open() as source:
            valid = np.array_equal(source.read(1), data)
            valid = valid and source.crs.to_epsg() == 31370
    return report("raster read and write", valid)


def check_project() -> bool:
    """Calculate the Sun's position through the project package."""
    from eclipse_viewshed import solar

    altitude, azimuth = solar.sun_altaz(18.2)
    valid = abs(altitude - 7.8) < 0.5 and abs(azimuth - 284.0) < 1.0
    return report("solar calculation", valid, f"alt={altitude:.2f}, az={azimuth:.2f}")


if __name__ == "__main__":
    checks = [
        check_imports(),
        check_gdal(),
        check_transform(),
        check_raster(),
        check_project(),
    ]
    raise SystemExit(0 if all(checks) else 1)
