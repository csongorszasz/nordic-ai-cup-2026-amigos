"""Tests for the NLS orthophoto sheet converter."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from offline.convert_nls_ortho import (
    SheetInfo,
    build_manifest,
    convert_sheet,
    find_jp2_sheets,
    read_georeference,
)


def _make_sheet(tmp_path: Path, year: str, sheet50k: str, sheet_id: str, letter: str, size: int = 64) -> Path:
    """Create a miniature NLS-style File-service tree with one JP2 sheet."""
    path = (
        tmp_path / "finland" / "orto" / "etrs-tm35fin" / "mara_v_25000_50"
        / year / sheet50k / "02m" / "1" / f"{sheet_id}{letter}.jp2"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(path, format="JPEG2000")
    return path


def test_sheet_info_parses_the_file_service_path(tmp_path):
    path = _make_sheet(tmp_path, "2026", "L34", "L3413", "C")
    info = SheetInfo.from_path(path)

    assert info is not None
    assert info.year == "2026"
    assert info.sheet50k == "L34"
    assert info.sheet_id == "L3413"
    assert info.letter == "C"


def test_sheet_info_rejects_unexpected_names(tmp_path):
    stray = tmp_path / "finland" / "something_else.jp2"
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"not a sheet")
    assert SheetInfo.from_path(stray) is None


def test_find_jp2_sheets_discovers_all_years_and_tiles(tmp_path):
    _make_sheet(tmp_path, "2026", "L34", "L3413", "A")
    _make_sheet(tmp_path, "2026", "L34", "L3413", "B")
    _make_sheet(tmp_path, "2023", "L41", "L4131", "F")

    sheets = find_jp2_sheets(tmp_path / "finland")

    # Paths sort by year first (2023 < 2026).
    assert [(s.sheet_id + s.letter, s.year) for s in sheets] == [
        ("L4131F", "2023"),
        ("L3413A", "2026"),
        ("L3413B", "2026"),
    ]


def test_convert_sheet_writes_usable_rgb_output(tmp_path):
    path = _make_sheet(tmp_path, "2026", "L34", "L3413", "C", size=64)
    info = SheetInfo.from_path(path)
    out_dir = tmp_path / "out"

    entry = convert_sheet(info, out_dir, fmt="jpg", quality=90, gsd_override=0.5)

    assert not entry["skipped"]
    out = out_dir / "2026" / "L3413C.jpg"
    assert out.is_file()
    with Image.open(out) as result:
        assert result.format == "JPEG"
        assert result.mode == "RGB"
        assert result.size == (64, 64)
    assert entry["gsd_m"] == 0.5
    assert entry["gsd_source"] == "explicit-override"
    assert entry["tile_size_m"] == pytest.approx(32)


def test_convert_sheet_is_idempotent(tmp_path):
    path = _make_sheet(tmp_path, "2026", "L34", "L3413", "C")
    info = SheetInfo.from_path(path)
    out_dir = tmp_path / "out"

    first = convert_sheet(info, out_dir, fmt="jpg", quality=90, gsd_override=0.5)
    second = convert_sheet(info, out_dir, fmt="jpg", quality=90, gsd_override=0.5)

    assert not first["skipped"]
    assert second["skipped"]
    assert second["width_px"] == first["width_px"]
    assert build_manifest([second])["gsd_m"] == 0.5


def test_manifest_summarises_the_converted_tree(tmp_path):
    path = _make_sheet(tmp_path, "2026", "L34", "L3413", "C")
    info = SheetInfo.from_path(path)
    entry = convert_sheet(info, tmp_path / "out", fmt="jpg", quality=90, gsd_override=0.5)

    manifest = build_manifest([entry])

    assert manifest["provider"].startswith("Maanmittauslaitos")
    assert manifest["gsd_m"] == 0.5
    assert manifest["years"] == ["2026"]
    assert manifest["pixel_sizes"] == [64]
    assert len(manifest["tiles"]) == 1


def _box(kind, contents):
    return struct.pack(">I4s", len(contents) + 8, kind) + contents


def _gml(scale=0.5):
    return f'''<gml:FeatureCollection xmlns:gml="http://www.opengis.net/gml">
    <gml:RectifiedGrid dimension="2">
      <gml:origin><gml:Point srsName="urn:ogc:def:crs:EPSG::3067">
        <gml:pos>476000.25 6851999.75</gml:pos>
      </gml:Point></gml:origin>
      <gml:offsetVector srsName="urn:ogc:def:crs:EPSG::3067">{scale} 0</gml:offsetVector>
      <gml:offsetVector srsName="urn:ogc:def:crs:EPSG::3067">0 -{scale}</gml:offsetVector>
    </gml:RectifiedGrid></gml:FeatureCollection>'''.encode()


@pytest.mark.parametrize("terminator", [b"", b"\x00"])
def test_georeference_after_codestream_is_authoritative(tmp_path, terminator):
    path = tmp_path / "tile.jp2"
    path.write_bytes(_box(b"jp2c", b"image-data" * 1000) + _box(b"asoc", _box(b"xml ", _gml() + terminator)))
    metadata = read_georeference(path)
    assert metadata["gsd_m"] == 0.5
    assert metadata["gsd_source"] == "embedded-gml"
    assert metadata["origin_pixel_center"] == [476000.25, 6851999.75]


def test_missing_georeference_requires_explicit_override(tmp_path):
    path = _make_sheet(tmp_path, "2023", "M44", "M4444", "A")
    with pytest.raises(ValueError, match="provide --gsd-m"):
        convert_sheet(SheetInfo.from_path(path), tmp_path / "out", "jpg", 90)


def test_contradictory_override_is_rejected_before_decoding(tmp_path):
    path = tmp_path / "tile.jp2"
    path.write_bytes(_box(b"xml ", _gml()))
    info = SheetInfo(path, "2023", "M44", "M4444", "A")
    with pytest.raises(ValueError, match="contradicts"):
        convert_sheet(info, tmp_path / "out", "jpg", 90, gsd_override=0.25)


def test_invalid_box_lengths_are_not_silently_ignored(tmp_path):
    path = tmp_path / "bad.jp2"
    path.write_bytes(struct.pack(">I4s", 1000, b"xml ") + b"short")
    with pytest.raises(ValueError, match="box length"):
        read_georeference(path)
