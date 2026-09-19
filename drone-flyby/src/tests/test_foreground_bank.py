import numpy as np
import pytest

from offline.extract_foreground_bank import alpha_asset, checker_preview


def test_alpha_asset_carries_only_masked_foreground_and_exact_bounds():
    image = np.full((10, 12, 3), 200, dtype=np.uint8)
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:8, 3:9] = True
    asset, box = alpha_asset(image, mask)
    assert box == (3, 2, 9, 8)
    assert asset.shape == (10, 12, 4)
    assert (asset[:, :, 3] > 0).sum() == 36
    preview = checker_preview(asset)
    assert tuple(preview[3, 4]) == (200, 200, 200)
    assert tuple(preview[0, 0]) != (200, 200, 200)


def test_empty_or_misaligned_masks_fail_explicitly():
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="empty"):
        alpha_asset(image, np.zeros((10, 12), dtype=bool))
    with pytest.raises(ValueError, match="align"):
        alpha_asset(image, np.ones((12, 10), dtype=bool))
