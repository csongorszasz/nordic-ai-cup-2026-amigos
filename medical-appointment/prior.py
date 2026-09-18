"""Question-text prior: a guess at yes/no from the wording alone.

The fallback when the expensive path fails (ASR error, LLM error, unparseable
reply, deadline). A constant answer is worth exactly 0.5 on a balanced set; the
question text alone does better, mostly because off-topic questions read
differently ("Is there any mention of attending a concert?").

Character 3–5-gram + word features, logistic regression in plain numpy, trained
at import from ``data/question_train.csv`` (390 rows, well under a second). No
sklearn, so it runs in any serving env.

    python prior.py          # leave-one-conversation-out accuracy by type
"""

import csv
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

QUESTIONS = Path(__file__).resolve().parent / "data" / "question_train.csv"

_WORD_RE = re.compile(r"[a-z0-9]+")
_MIN_COUNT = 2


def _features(question: str) -> List[str]:
    text = f" {' '.join(_WORD_RE.findall(question.lower()))} "
    grams = [
        f"c{n}:{text[i:i + n]}"
        for n in (3, 4, 5)
        for i in range(len(text) - n + 1)
    ]
    words = text.split()
    grams += [f"w:{word}" for word in words]
    grams += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
    return grams


class QuestionPrior:
    """Binary logistic regression over hashed-free sparse n-gram features."""

    def __init__(self, epochs: int = 300, lr: float = 0.5, l2: float = 1e-3) -> None:
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2
        self.vocab: Dict[str, int] = {}
        self.weights = None
        self.bias = 0.0

    def _matrix(self, questions: Sequence[str]):
        import numpy as np

        matrix = np.zeros((len(questions), len(self.vocab)), dtype=np.float32)
        for row, question in enumerate(questions):
            for gram in _features(question):
                column = self.vocab.get(gram)
                if column is not None:
                    matrix[row, column] = 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-6)

    def fit(self, questions: Sequence[str], labels: Sequence[int]) -> "QuestionPrior":
        import numpy as np

        counts = Counter(gram for q in questions for gram in set(_features(q)))
        self.vocab = {
            gram: index
            for index, gram in enumerate(sorted(g for g, c in counts.items() if c >= _MIN_COUNT))
        }
        x = self._matrix(questions)
        y = np.asarray(labels, dtype=np.float32)
        self.weights = np.zeros(x.shape[1], dtype=np.float32)
        self.bias = 0.0
        for _ in range(self.epochs):
            p = 1.0 / (1.0 + np.exp(-(x @ self.weights + self.bias)))
            grad = p - y
            self.weights -= self.lr * (x.T @ grad / len(y) + self.l2 * self.weights)
            self.bias -= self.lr * float(grad.mean())
        return self

    def proba(self, questions: Sequence[str]) -> List[float]:
        if self.weights is None:
            return [0.5] * len(questions)
        logits = self._matrix(questions) @ self.weights + self.bias
        return [1.0 / (1.0 + math.exp(-float(z))) for z in logits]

    def predict(self, questions: Sequence[str]) -> List[bool]:
        return [p >= 0.5 for p in self.proba(questions)]


def _load_rows(path: Path = QUESTIONS) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


_PRIOR: Optional[QuestionPrior] = None
_FAILED = False


def get_prior() -> Optional[QuestionPrior]:
    """The process-wide prior, trained once; ``None`` if the data is missing."""
    global _PRIOR, _FAILED
    if _PRIOR is None and not _FAILED:
        try:
            rows = _load_rows()
            _PRIOR = QuestionPrior().fit(
                [row["question"] for row in rows], [int(row["label"]) for row in rows]
            )
        except Exception:
            _FAILED = True
            logger.exception("question prior unavailable; fallback guesses are constant.")
    return _PRIOR


def guess(questions: Sequence[str], default: bool = True) -> List[bool]:
    """Best cheap guess for each question. Never raises."""
    try:
        prior = get_prior()
        if prior is not None:
            return prior.predict(questions)
    except Exception:
        logger.exception("question prior failed; using the constant guess.")
    return [default] * len(questions)


def loco_report(rows: Optional[List[Dict[str, str]]] = None) -> Dict:
    """Leave-one-conversation-out accuracy, overall and by question type."""
    rows = rows if rows is not None else _load_rows()
    by_tid: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_tid[row["transcript_id"]].append(row)
    correct: Counter = Counter()
    total: Counter = Counter()
    for tid, held_out in by_tid.items():
        train = [row for row in rows if row["transcript_id"] != tid]
        model = QuestionPrior().fit(
            [r["question"] for r in train], [int(r["label"]) for r in train]
        )
        predictions = model.predict([r["question"] for r in held_out])
        for row, prediction in zip(held_out, predictions):
            hit = int(prediction == bool(int(row["label"])))
            for key in ("all", row["question_type"]):
                correct[key] += hit
                total[key] += 1
    return {key: round(correct[key] / total[key], 4) for key in total}


if __name__ == "__main__":
    for key, value in sorted(loco_report().items()):
        print(f"{key:<15}{value:.3f}")
