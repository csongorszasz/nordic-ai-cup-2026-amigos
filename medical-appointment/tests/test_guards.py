"""Numeric guard: veto rules and number extraction."""

from guards import numbers_in, numeric_guard, vetoed


def test_extracts_digits_and_decimals():
    assert numbers_in("The dose is 100 mg daily.") == {100.0}
    assert numbers_in("HbA1c came out at 47 mmol/mol") == {47.0}
    assert numbers_in("0.3 mg once a day") == {0.3, 1.0}


def test_extracts_number_words():
    assert 2.0 in numbers_in("for two weeks")
    assert 6.0 in numbers_in("six weeks")


def test_alphanumeric_codes_are_not_numbers():
    assert numbers_in("COVID-19 exposure") == set()
    assert numbers_in("HbA1c of 47") == {47.0}


def test_guard_vetoes_mismatched_number():
    assert vetoed("Was the prescribed dose 200 mg daily?",
                  "Sporanox 100 mg daily for 2 weeks.")


def test_guard_allows_matching_number():
    assert numeric_guard("Should the daily dose be 100 mg?",
                         "Sporanox 100 mg daily for 2 weeks.")


def test_guard_passes_when_question_has_no_number():
    assert numeric_guard("Did the patient report new symptoms?",
                         "Nothing new was reported.")


def test_guard_spoken_versus_digits():
    assert numeric_guard("Will the treatment last two weeks?",
                         "Continue for 2 weeks.")
