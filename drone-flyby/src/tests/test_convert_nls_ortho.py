"""Tests for the NLS orthophoto sheet converter."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from offline.convert_nls_ortho import (
    INFERRED_GSD,
    SheetInfo,
    build_manifest,
    convert_sheet,
    find_jp2_sheets,
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

    entry = convert_sheet(info, out_dir, fmt="jpg", quality=90)

    assert not entry["skipped"]
    out = out_dir / "2026" / "L3413C.jpg"
    assert out.is_file()
    with Image.open(out) as result:
        assert result.format == "JPEG"
        assert result.mode == "RGB"
        assert result.size == (64, 64)
    assert entry["gsd_m"] == INFERRED_GSD
    assert entry["gsd_source"] == "inferred"
    assert entry["tile_size_m"] == pytest.approx(64 * INFERRED_GSD)


def test_convert_sheet_is_idempotent(tmp_path):
    path = _make_sheet(tmp_path, "2026", "L34", "L3413", "C")
    info = SheetInfo.from_path(path)
    out_dir = tmp_path / "out"

    first = convert_sheet(info, out_dir, fmt="jpg", quality=90)
    second = convert_sheet(info, out_dir, fmt="jpg", quality=90)

    assert not first["skipped"]
    assert second["skipped"]


def test_manifest_summarises_the_converted_tree(tmp_path):
    path = _make_sheet(tmp_path, "2026", "L34", "L3413", "C")
    info = SheetInfo.from_path(path)
    entry = convert_sheet(info, tmp_path / "out", fmt="jpg", quality=90)

    manifest = build_manifest([entry])

    assert manifest["provider"].startswith("Maanmittauslaitos")
    assert manifest["gsd_m"] == INFERRED_GSD
    assert manifest["years"] == ["2026"]
    assert manifest["pixel_sizes"] == [64]
    assert len(manifest["tiles"]) == 1
