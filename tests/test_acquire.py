"""Tests for WCS request building and response unwrapping.

No network. The multipart handling is tested against a synthetic response
matching what geo.api.vlaanderen.be actually returned on the first run, so the
parser is pinned to observed behaviour rather than to the spec.

Run with:  pytest -v
"""


import pytest


from eclipse_viewshed import acquire, aoi

# Minimal but valid-enough GeoTIFF header: 'II', version 42, offset to IFD.
FAKE_TIFF = b"II\x2a\x00\x08\x00\x00\x00" + b"\x00" * 64
FAKE_TIFF_BE = b"MM\x00\x2a\x00\x00\x00\x08" + b"\x00" * 64


def multipart_response(tiff: bytes = FAKE_TIFF) -> tuple[bytes, str]:
    """Reproduce the shape of the service's actual reply: a GML part followed
    by the raster part, boundary 'wcs', as seen in the first notebook run."""
    body = (
        b"\r\n--wcs\r\n"
        b"Content-Type: text/xml\r\n"
        b"Content-ID: GML-Part\r\n\r\n"
        b'<?xml version="1.0" encoding="UTF-8"?>\r\n'
        b"<gmlcov:RectifiedGridCoverage/>\r\n"
        b"\r\n--wcs\r\n"
        b"Content-Type: image/tiff\r\n"
        b"Content-ID: coverage.tif\r\n"
        b"Content-Transfer-Encoding: binary\r\n\r\n"
        + tiff +
        b"\r\n--wcs--\r\n"
    )
    ctype = 'multipart/related; boundary="wcs";start="GML-Part";type="text/xml"'
    return body, ctype


# ------------------------------------------------------- response unwrapping

def test_bare_geotiff_passes_through():
    assert acquire.extract_geotiff(FAKE_TIFF, "image/tiff") == FAKE_TIFF


def test_big_endian_geotiff_accepted():
    assert acquire.extract_geotiff(FAKE_TIFF_BE, "image/tiff") == FAKE_TIFF_BE


def test_multipart_is_unwrapped():
    """The failure mode from the first run: WCS 2.0.1 wraps the raster in a
    multipart message with a GML description as the first part."""
    body, ctype = multipart_response()
    assert acquire.extract_geotiff(body, ctype) == FAKE_TIFF


def test_multipart_result_is_openable_shape():
    body, ctype = multipart_response()
    out = acquire.extract_geotiff(body, ctype)
    assert out[:2] in acquire.TIFF_MAGIC
    assert not out.startswith(b"\r\n--wcs")


def test_multipart_without_raster_raises():
    body = (b"\r\n--wcs\r\nContent-Type: text/xml\r\n\r\n<gml:Stuff/>\r\n"
            b"\r\n--wcs--\r\n")
    with pytest.raises(RuntimeError, match="no GeoTIFF part"):
        acquire.extract_geotiff(body, 'multipart/related; boundary="wcs"')


def test_xml_exception_raises_with_server_text():
    """WCS reports errors as XML with HTTP 200, so status codes alone are not
    enough to detect failure."""
    err = (b'<?xml version="1.0"?><ows:ExceptionReport>'
           b'<ows:Exception exceptionCode="InvalidSubsetting"/>'
           b'</ows:ExceptionReport>')
    with pytest.raises(RuntimeError, match="InvalidSubsetting"):
        acquire.extract_geotiff(err, "text/xml")


def test_error_message_includes_content_type():
    with pytest.raises(RuntimeError, match="text/html"):
        acquire.extract_geotiff(b"<html>502 Bad Gateway</html>", "text/html")


# ---------------------------------------------------------- URL construction

def test_coverage_url_has_required_wcs_parameters():
    url = acquire.coverage_url(acquire.COVERAGE_DSM,
                               aoi.Bounds(151000, 211000, 152000, 212000))
    for token in ("service=WCS", "version=2.0.1", "request=GetCoverage",
                  "coverageId=DHMVII_DSM_1m", "format=image/tiff"):
        assert token in url


def test_coverage_url_subsets_both_axes():
    url = acquire.coverage_url(acquire.COVERAGE_DSM,
                               aoi.Bounds(151000, 211000, 152000, 212000))
    assert "subset=x(151000,152000)" in url
    assert "subset=y(211000,212000)" in url


def test_coverage_url_respects_axis_labels():
    """Axis names come from DescribeCoverage; a server using E/N instead of
    x/y would need different subset parameters."""
    url = acquire.coverage_url(acquire.COVERAGE_DSM,
                               aoi.Bounds(151000, 211000, 152000, 212000),
                               axis_labels=("E", "N"))
    assert "subset=E(151000,152000)" in url
    assert "subset=N(211000,212000)" in url


def test_coverage_url_matches_tile_bounds():
    key = (151000, 211000)
    url = acquire.coverage_url(acquire.COVERAGE_DTM, aoi.tile_bounds(key, 1000))
    assert "subset=x(151000,152000)" in url
    assert "DHMVII_DTM_1m" in url


# ----------------------------------------------------------------- caching

def test_fetch_tile_returns_cached_without_network(tmp_path):
    """A tile already on disk must not trigger a request. This is what makes
    adding a second observation point cheap."""
    key = (151000, 211000)
    existing = tmp_path / f"{acquire.COVERAGE_DSM}_{aoi.tile_name(key)}.tif"
    existing.write_bytes(FAKE_TIFF)

    path, downloaded = acquire.fetch_tile(acquire.COVERAGE_DSM, key, tmp_path)

    assert downloaded is False
    assert path == existing


def test_fetch_tile_ignores_empty_cache_file(tmp_path, monkeypatch):
    """A zero-byte file from an interrupted run must not be treated as cached."""
    key = (151000, 211000)
    stub = tmp_path / f"{acquire.COVERAGE_DSM}_{aoi.tile_name(key)}.tif"
    stub.write_bytes(b"")

    called = {"n": 0}

    class FakeResponse:
        content = FAKE_TIFF
        headers = {"Content-Type": "image/tiff"}

        def raise_for_status(self):
            pass

    def fake_get(url, timeout=None):
        called["n"] += 1
        return FakeResponse()

    monkeypatch.setattr(acquire.requests, "get", fake_get)
    path, downloaded = acquire.fetch_tile(acquire.COVERAGE_DSM, key, tmp_path)

    assert called["n"] == 1
    assert downloaded is True
    assert path.read_bytes() == FAKE_TIFF


def test_fetch_tile_writes_unwrapped_multipart(tmp_path, monkeypatch):
    """End to end: a multipart reply must land on disk as a bare GeoTIFF."""
    body, ctype = multipart_response()

    class FakeResponse:
        content = body
        headers = {"Content-Type": ctype}

        def raise_for_status(self):
            pass

    monkeypatch.setattr(acquire.requests, "get", lambda url, timeout=None: FakeResponse())
    path, downloaded = acquire.fetch_tile(acquire.COVERAGE_DSM, (151000, 211000), tmp_path)

    assert downloaded is True
    assert path.read_bytes() == FAKE_TIFF


def test_ensure_tiles_collects_failures_without_aborting(tmp_path, monkeypatch):
    """One bad tile must not lose a long run."""
    keys = [(151000, 211000), (152000, 211000), (153000, 211000)]

    def flaky(coverage_id, key, outdir, *a, **kw):
        if key == keys[1]:
            raise RuntimeError("simulated server error")
        p = outdir / f"{aoi.tile_name(key)}.tif"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(FAKE_TIFF)
        return p, True

    monkeypatch.setattr(acquire, "fetch_tile", flaky)
    report = acquire.ensure_tiles(acquire.COVERAGE_DSM, keys, tmp_path,
                                  pause_s=0, progress=False)

    assert report.requested == 3
    assert report.downloaded == 2
    assert len(report.failed) == 1
    assert report.failed[0][0] == keys[1]


# ------------------------------- regression: runtime-resolved configuration

def test_fetch_tile_resolves_tile_m_at_call_time(tmp_path, monkeypatch):
    """acquire must follow aoi.TILE_M rather than a value frozen at import.

    Regression test. acquire.py originally did `from .aoi import TILE_M`, which
    binds a copy, and then used it as a signature default. Changing aoi.TILE_M
    left acquire writing tiles at the old size, so one cache directory could end
    up holding tiles of two different grids.
    """
    monkeypatch.setattr(aoi, "TILE_M", 500)

    class FakeResponse:
        content = FAKE_TIFF
        headers = {"Content-Type": "image/tiff"}

        def raise_for_status(self):
            pass

    monkeypatch.setattr(acquire.requests, "get",
                        lambda url, timeout=None: FakeResponse())
    path, _ = acquire.fetch_tile(acquire.COVERAGE_DSM, (151000, 211000), tmp_path)
    assert path.name.endswith("_t500.tif")


def test_coverage_url_spans_one_tile_width(monkeypatch):
    """The requested subset must match the active tile size."""
    monkeypatch.setattr(aoi, "TILE_M", 500)
    url = acquire.coverage_url(acquire.COVERAGE_DSM, aoi.tile_bounds((151000, 211000)))
    assert "subset=x(151000,151500)" in url
