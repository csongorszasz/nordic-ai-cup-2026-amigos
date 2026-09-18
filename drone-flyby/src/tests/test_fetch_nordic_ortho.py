"""Tests for the Danish orthophoto strip fetcher (offline, HTTP mocked)."""

import json
import math

import cv2
import numpy as np
import pytest
from scipy import integrate

import offline.fetch_nordic_ortho as fetcher
from offline.fetch_nordic_ortho import (
    _A,
    _E2,
    _K0,
    StripSpec,
    fetch_strip,
    iter_target_blocks,
    latlon_to_utm32n,
)


# --------------------------------------------------------------------------- #
# UTM 32N conversion, cross-checked against independent computations
# --------------------------------------------------------------------------- #

def test_central_meridian_maps_to_false_easting():
    easting, _ = latlon_to_utm32n(55.7403, 9.0)
    assert easting == pytest.approx(500000.0, abs=1e-6)


def test_small_longitude_offset_matches_the_closed_form():
    lat, dlon_deg = 55.7403, 0.01
    easting, _ = latlon_to_utm32n(lat, 9.0 + dlon_deg)
    phi = math.radians(lat)
    n = _A / math.sqrt(1.0 - _E2 * math.sin(phi) ** 2)
    expected = 500000.0 + _K0 * n * math.cos(phi) * math.radians(dlon_deg)
    assert easting == pytest.approx(expected, abs=1.0)


def test_northing_matches_numerical_meridian_arc():
    lat = 55.7403
    _, northing = latlon_to_utm32n(lat, 9.0)

    def meridian_radius(phi):
        return _A * (1.0 - _E2) / (1.0 - _E2 * math.sin(phi) ** 2) ** 1.5

    arc, _ = integrate.quad(meridian_radius, 0.0, math.radians(lat))
    assert northing == pytest.approx(_K0 * arc, abs=0.5)


# --------------------------------------------------------------------------- #
# Target-block tiling
# --------------------------------------------------------------------------- #

def test_blocks_cover_every_target_pixel_exactly_once():
    coverage = np.zeros((13, 29), dtype=np.int32)
    for x0, y0, w, h in iter_target_blocks(29, 13, 8):
        coverage[y0 : y0 + h, x0 : x0 + w] += 1
    assert set(np.unique(coverage)) == {1}


def test_strip_spec_bbox_and_target_size():
    spec = StripSpec(
        site="test", center_e=500000.0, center_n=6175000.0,
        length_m=4000.0, width_m=1000.0, direction="we",
    )
    e0, n0, e1, n1 = spec.bbox()
    assert (e1 - e0) == pytest.approx(4000.0)
    assert (n1 - n0) == pytest.approx(1000.0)
    assert spec.target_size() == (round(4000.0 / fetcher.TARGET_GSD), round(1000.0 / fetcher.TARGET_GSD))

    spec_ns = StripSpec(
        site="test", center_e=500000.0, center_n=6175000.0,
        length_m=4000.0, width_m=1000.0, direction="ns",
    )
    _, _, e1_ns, n1_ns = spec_ns.bbox()
    assert (n1_ns - 6175000.0 + 2000.0) == pytest.approx(4000.0)
    assert (e1_ns - 500000.0 + 500.0) == pytest.approx(1000.0)


# --------------------------------------------------------------------------- #
# End-to-end fetch with a mocked WMS
# --------------------------------------------------------------------------- #

class _FakeWMS:
    """Serves synthetic images sized by the WIDTH/HEIGHT in the request URL."""

    def __init__(self, seed_value: int = 7):
        self.seed_value = seed_value
        self.requests: list[dict] = []

    def __call__(self, url: str, timeout: float = 60.0) -> bytes:
        params = dict(p.split("=", 1) for p in url.split("?", 1)[1].split("&"))
        self.requests.append(params)
        width, height = int(params["width"]), int(params["height"])
        image = np.full((height, width, 3), self.seed_value, dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", image)
        assert ok
        return buffer.tobytes()


def test_fetch_strip_end_to_end_with_mock(tmp_path, monkeypatch):
    fake = _FakeWMS(seed_value=200)
    monkeypatch.setattr(fetcher, "_http_get_bytes", fake)

    spec = StripSpec(
        site="mock", center_e=505000.0, center_n=6175000.0,
        length_m=240.0, width_m=120.0, direction="we",
    )
    image_path = fetch_strip(spec, token="dummy", output_dir=tmp_path, block_px=96, request_delay_s=0.0)

    assert image_path.is_file()
    metadata = json.loads((tmp_path / "strip.json").read_text())
    assert metadata["crs"] == fetcher.CRS
    assert metadata["flight_direction"] == "we"
    assert metadata["n_requests"] == len(fake.requests)
    assert metadata["license"].startswith("Open data")

    strip = cv2.imread(str(image_path))
    width, height = spec.target_size()
    assert strip.shape == (height, width, 3)
    # Every target pixel was written by exactly one mocked block.
    assert int(strip.min()) == 200 and int(strip.max()) == 200


def test_request_block_urls_ask_for_native_resolution(tmp_path, monkeypatch):
    fake = _FakeWMS()
    monkeypatch.setattr(fetcher, "_http_get_bytes", fake)

    spec = StripSpec(site="mock", center_e=500000.0, center_n=6175000.0,
                     length_m=240.0, width_m=120.0, direction="we")
    block = next(iter_target_blocks(*spec.target_size(), 120))
    fetcher._request_block(spec, block, token="dummy", timeout=1.0)

    params = fake.requests[0]
    from urllib.parse import unquote

    assert unquote(params["srs"]) == fetcher.CRS
    assert unquote(params["layers"]) == spec.layer
    assert int(params["width"]) == round(120 * fetcher.TARGET_GSD / fetcher.NATIVE_GSD)
    assert unquote(params["token"]) == "dummy"


class _NorthGradientWMS:
    """Paints every returned row by its northing, so orientation is testable.

    WMS GetMap returns ``maxy`` on the top image row; the mock reproduces that
    contract from the bbox it receives.
    """

    def __init__(self, south: float, north: float, native_gsd: float):
        self.south = south
        self.span = north - south
        self.native_gsd = native_gsd
        self.requests: list[dict] = []

    def __call__(self, url: str, timeout: float = 60.0) -> bytes:
        from urllib.parse import unquote

        params = dict(p.split("=", 1) for p in url.split("?", 1)[1].split("&"))
        self.requests.append(params)
        bbox = [float(v) for v in unquote(params["bbox"]).split(",")]
        _, miny, _, maxy = bbox
        width, height = int(params["width"]), int(params["height"])
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for row in range(height):
            northing = maxy - (row + 0.5) * self.native_gsd
            value = int(round((northing - self.south) / self.span * 255))
            image[row, :] = np.clip(value, 0, 255)
        ok, buffer = cv2.imencode(".png", image)
        assert ok
        return buffer.tobytes()


def test_fetch_strip_is_north_up(tmp_path, monkeypatch):
    spec = StripSpec(
        site="orientation", center_e=500000.0, center_n=6175000.0,
        length_m=240.0, width_m=120.0, direction="we",
    )
    e0, n0, _, n1 = spec.bbox()
    fake = _NorthGradientWMS(south=n0, north=n1, native_gsd=fetcher.NATIVE_GSD)
    monkeypatch.setattr(fetcher, "_http_get_bytes", fake)

    fetch_strip(spec, token="dummy", output_dir=tmp_path, block_px=256, request_delay_s=0.0)

    strip = cv2.imread(str(tmp_path / "strip.png"))
    height = spec.target_size()[1]
    gsd = spec.target_gsd
    expected_top = (n1 - 0.5 * gsd - n0) / (n1 - n0) * 255.0
    expected_bottom = (n1 - (height - 0.5) * gsd - n0) / (n1 - n0) * 255.0

    assert strip[0, :, 0].mean() == pytest.approx(expected_top, abs=3.0)
    assert strip[-1, :, 0].mean() == pytest.approx(expected_bottom, abs=3.0)
    # North must decrease monotonically down the strip.
    row_means = strip[:, :, 0].mean(axis=1)
    assert np.all(np.diff(row_means) <= 1e-6)

    metadata = json.loads((tmp_path / "strip.json").read_text())
    assert metadata["north_up"] is True


def test_fix_existing_flips_only_unmarked_strips(tmp_path):
    south_up = np.zeros((10, 8, 3), dtype=np.uint8)
    south_up[0, :] = 255  # something at the "top"
    (tmp_path / "old_site").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "old_site" / "strip.png"), south_up)
    (tmp_path / "old_site" / "strip.json").write_text(json.dumps({"site": "old_site"}))
    (tmp_path / "new_site").mkdir()
    cv2.imwrite(str(tmp_path / "new_site" / "strip.png"), south_up.copy())
    (tmp_path / "new_site" / "strip.json").write_text(json.dumps({"site": "new_site", "north_up": True}))

    fixed = fetcher.fix_existing_strips(tmp_path)

    assert fixed == 1
    flipped = cv2.imread(str(tmp_path / "old_site" / "strip.png"))
    assert flipped[0, 0, 0] == 0 and flipped[-1, 0, 0] == 255  # was flipped
    assert json.loads((tmp_path / "old_site" / "strip.json").read_text())["north_up"] is True
    untouched = cv2.imread(str(tmp_path / "new_site" / "strip.png"))
    assert untouched[0, 0, 0] == 255  # already marked, untouched


def test_missing_token_is_handled(capsys, monkeypatch):
    monkeypatch.delenv(fetcher.TOKEN_ENV, raising=False)
    monkeypatch.setattr("sys.argv", ["prog"])
    assert fetcher.main() == 1
    assert fetcher.TOKEN_ENV in capsys.readouterr().out
