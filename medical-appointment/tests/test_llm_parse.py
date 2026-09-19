"""LLM output parsing: JSON extraction and answer normalisation."""

from answerers.llm_parse import extract_json, parse_answers, parse_decisions

IDS = ["q01", "q02", "q03"]


def test_parse_plain_object():
    text = '{"answers":[{"id":"q01","answer":"yes","evidence_quote":"abc"},' \
           '{"id":"q02","answer":"no","evidence_quote":null}]}'
    parsed = parse_answers(text, IDS)
    assert parsed["q01"] == {"answer": True, "quote": "abc", "candidate": None}
    assert parsed["q02"] == {"answer": False, "quote": None, "candidate": None}
    assert parsed["q03"] is None


def test_parse_candidate_id():
    text = ('{"answers":[{"id":"q01","answer":"yes","candidate":"c03",'
            '"evidence_quote":"abc"}]}')
    parsed = parse_answers(text, IDS)
    assert parsed["q01"]["candidate"] == "c03"


def test_parse_fenced_and_prose():
    text = 'Sure!\n```json\n{"answers":[{"id":"q01","answer":"true"}]}\n```'
    assert parse_answers(text, IDS)["q01"]["answer"] is True


def test_parse_bare_list_and_boolean():
    text = '[{"id":"q01","answer":true},{"id":"q02","answer":false}]'
    parsed = parse_answers(text, IDS)
    assert parsed["q01"]["answer"] is True
    assert parsed["q02"]["answer"] is False


def test_parse_garbage_returns_all_none():
    assert parse_answers("not json at all", IDS) == {"q01": None, "q02": None, "q03": None}


def test_extract_json_ignores_trailing_text():
    assert extract_json('prefix {"a": 1} suffix') == {"a": 1}


def test_parse_decisions():
    text = '{"answers":[{"id":"q01","answer":"yes"},{"id":"q02","answer":"no"}]}'
    decisions = parse_decisions(text, IDS)
    assert decisions == {"q01": True, "q02": False, "q03": None}


def test_truncated_json_preserves_complete_slots_only():
    text = (
        '{"answers":[{"id":"q01","answer":"yes","evidence_quote":"A } quote"},'
        '{"id":"q02","answer":"no"},'
        '{"id":"q03","answer":"ye'
    )
    parsed = parse_answers(text, IDS)
    assert parsed["q01"]["quote"] == "A } quote"
    assert parsed["q02"]["answer"] is False
    assert parsed["q03"] is None


def test_wrong_answer_container_does_not_raise():
    assert parse_answers('{"answers":17}', IDS) == dict.fromkeys(IDS)


def test_duplicate_id_is_not_silently_overwritten():
    text = '{"answers":[{"id":"q01","answer":"yes"},{"id":"q01","answer":"no"}]}'
    assert parse_answers(text, IDS)["q01"] is None


def test_parse_quotes_collects_candidate_lists():
    from answerers.llm_parse import parse_quotes

    text = ('{"answers":[{"id":"q01","answer":"yes","evidence_quotes":["a b","c d"]},'
            '{"id":"q02","answer":"no","evidence_quotes":[]}]}')
    parsed = parse_quotes(text, ["q01", "q02"])
    assert parsed["q01"] == {"answer": True, "quotes": ["a b", "c d"]}
    assert parsed["q02"] == {"answer": False, "quotes": []}


def test_parse_quotes_accepts_single_string():
    from answerers.llm_parse import parse_quotes

    parsed = parse_quotes('{"answers":[{"id":"q01","answer":"yes","evidence_quotes":"solo"}]}', ["q01"])
    assert parsed["q01"]["quotes"] == ["solo"]


def test_parse_reason_field_is_ignored():
    text = (
        '{"answers":[{"id":"q01","reason":"because X","answer":"yes",'
        '"evidence_quote":"abc"},{"id":"q02","reason":"none","answer":"no",'
        '"evidence_quote":null}]}'
    )
    parsed = parse_answers(text, IDS)
    assert parsed["q01"] == {"answer": True, "quote": "abc", "candidate": None}
    assert parsed["q02"] == {"answer": False, "quote": None, "candidate": None}


def test_truncated_json_with_reason_preserves_complete_slots():
    text = (
        '{"answers":[{"id":"q01","reason":"because X","answer":"yes",'
        '"evidence_quote":"abc"},{"id":"q02","reason":"ye'
    )
    parsed = parse_answers(text, IDS)
    assert parsed["q01"]["quote"] == "abc"
    assert parsed["q02"] is None
