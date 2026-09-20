"""Citation-confidence helpers: pure logprob math and entry segmentation."""

import numpy as np
import pytest

from answerers.llm_client import mean_token_logprob, token_logprobs
from confidence_probe import _ID_RE, segment_confidence, separation


def test_token_logprobs_uniform_and_peaked():
    logits = np.zeros((2, 4))
    values = token_logprobs(logits, [0, 1])
    assert values == pytest.approx([-np.log(4), -np.log(4)])
    peaked = np.array([[10.0, 0.0], [0.0, 10.0]])
    assert token_logprobs(peaked, [0, 1]) == pytest.approx([-0.0000454, -0.0000454], abs=1e-4)
    assert mean_token_logprob(peaked, [0, 1]) > mean_token_logprob(logits, [0, 1])


def test_token_logprobs_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        token_logprobs(np.zeros((2, 4)), [0])


class CharTokenizer:
    """One character per token, so prefixes are trivially decodable."""

    def __init__(self, text):
        self.text = text

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.text[i] for i in ids)


def test_segment_confidence_attributes_entries():
    text = '{"answers":[{"id":"q01","answer":"yes","evidence_quote":"a"},' \
           '{"id":"q02","answer":"no","evidence_quote":null}]}'
    tokenizer = CharTokenizer(text)
    token_ids = list(range(len(text)))
    logprobs = [0.0] * len(text)
    matches = list(_ID_RE.finditer(text))
    first, second = matches[0].start(), matches[1].start()
    for index in range(first, second):
        logprobs[index] = -2.0
    for index in range(second, len(text)):
        logprobs[index] = -1.0
    result = segment_confidence(tokenizer, token_ids, logprobs, text)
    assert result["q01"] == pytest.approx(-2.0)
    assert result["q02"] == pytest.approx(-1.0)


def test_separation_detects_order_and_ties():
    assert separation([-1, -1, -1], [-3, -3]) == 1.0
    assert separation([-3, -3], [-1, -1]) == 0.0
    assert separation([-2], [-2]) == 0.5
    assert separation([], [-1]) is None
