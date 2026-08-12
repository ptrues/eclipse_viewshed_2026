"""Eclipse viewshed analysis — Antwerp, 12 August 2026."""

__version__ = "0.1.0"

# `acquire` is imported explicitly by the notebooks rather than here, so that
# the package stays importable in environments without `requests` or GDAL.
from . import aoi, solar  # noqa: F401
