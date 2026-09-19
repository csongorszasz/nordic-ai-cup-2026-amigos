"""Medical ASR conversion must retain timestamp-alignment metadata."""

import pytest

from prepare_medical_whisper import verify_alignment_heads


def test_alignment_heads_must_survive_conversion_exactly():
    source = {"alignment_heads": [[7, 0], [10, 17]]}
    assert verify_alignment_heads(source, dict(source)) == 2
    for converted in ({}, {"alignment_heads": None}, {"alignment_heads": [[7, 0]]}):
        with pytest.raises(ValueError, match="alignment heads"):
            verify_alignment_heads(source, converted)


def test_missing_source_heads_do_not_silently_use_generic_defaults():
    with pytest.raises(ValueError, match="alignment heads"):
        verify_alignment_heads({}, {"alignment_heads": [[16, 0]]})
