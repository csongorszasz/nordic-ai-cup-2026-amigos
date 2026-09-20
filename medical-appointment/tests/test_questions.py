"""Table-driven tests for the question -> proposition rewrite."""

import pytest

from questions import strip_tag, to_proposition

CASES = [
    ("Should the daily dose be 100 mg?", "The daily dose should be 100 mg."),
    ("Was the prescribed dose 200 mg daily?",
     "The prescribed dose was 200 mg daily."),
    ("Will the treatment last two weeks?", "The treatment will last two weeks."),
    ("Is the medicine to be taken after a meal?",
     "The medicine is to be taken after a meal."),
    ("Were the lungs and heart normal on auscultation?",
     "The lungs and heart were normal on auscultation."),
    ("Did the patient report new breathing symptoms?",
     "The patient did report new breathing symptoms."),
    ("Has the doctor found the condition to be unstable?",
     "The doctor has found the condition to be unstable."),
    ("Is the skin intact?", "The skin is intact."),
    ("Is dry skin the reason for this consultation?",
     "Dry skin is the reason for this consultation."),
    ("Does the plan involve a check-up only every second year?",
     "The plan does involve a check-up only every second year."),
    ("The lipid profile came back normal, didn't it?",
     "The lipid profile came back normal."),
    ("The treatment is planned to run for six weeks, right?",
     "The treatment is planned to run for six weeks."),
]


@pytest.mark.parametrize("question,expected", CASES)
def test_to_proposition(question, expected):
    assert to_proposition(question) == expected


def test_strip_tag():
    assert strip_tag("The plan involves no treatment, right?") == \
        "The plan involves no treatment"
    assert strip_tag("The patient's thyroid function was checked, wasn't it?") == \
        "The patient's thyroid function was checked"


def test_proposition_keeps_numbers_and_entities():
    proposition = to_proposition("Should the fungal infection be treated with Brentan gel?")
    assert "Brentan" in proposition
    assert "fungal" in proposition


def test_proposition_is_idempotent_on_statements():
    statement = "The patient has asthma."
    assert to_proposition(statement) == statement
