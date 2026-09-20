"""Fixed OOF fusion has no learned threshold or decision changes."""

from evaluate_adapter_agreement import agreed_span


def test_agreement_keeps_baseline_when_citations_disagree():
    assert agreed_span([1.0, 3.0], [10.0, 12.0], "adapter_on_agreement") == [1.0, 3.0]
    assert agreed_span(None, [1.0, 2.0], "adapter_on_agreement") is None
    assert agreed_span([1.0, 2.0], None, "midpoint_on_agreement") == [1.0, 2.0]


def test_fixed_agreement_rules_use_only_the_two_proposals():
    assert agreed_span([1.0, 3.0], [1.2, 3.2], "adapter_on_agreement") == [1.2, 3.2]
    assert agreed_span([1.0, 3.0], [1.2, 3.2], "midpoint_on_agreement") == [1.1, 3.1]
