"""LLM output parsing: JSON extraction and answer normalisation."""

from answerers.llm_parse import extract_json, parse_answers, parse_decisions

IDS = ["q01", "q02", "q03"]


def test_parse_plain_object():
    text = '{"answers":[{"id":"q01","answer":"yes","evidence_quote":"abc"},' \
           '{"id":"q02","answer":"no","evidence_quote":null}]}'
    parsed = parse_answers(text, IDS)
    assert parsed["q01"] == {"answer": True, "quote": "abc"}
    assert parsed["q02"] == {"answer": False, "quote": None}
    assert parsed["q03"] is None


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
