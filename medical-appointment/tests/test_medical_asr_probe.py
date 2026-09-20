"""Missing or invalid candidate word times are explicit diagnostic failures."""

from probe_medical_asr import word_time_errors


def test_valid_word_times_need_no_repair():
    assert word_time_errors({
        "duration": 2.0, "words": [{"word": "hello", "start": 0.1, "end": 0.8}],
    }) == []


def test_empty_nonfinite_and_out_of_audio_times_are_flagged():
    assert word_time_errors({"duration": 2.0, "words": []})
    for start, end in ((0.0, 2.1), (-0.1, 0.3), (1.0, 0.5), (float("nan"), 1.0)):
        errors = word_time_errors({
            "duration": 2.0, "words": [{"word": "hello", "start": start, "end": end}],
        })
        assert len(errors) == 1 and "reason" in errors[0]
