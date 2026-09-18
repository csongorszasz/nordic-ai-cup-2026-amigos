"""Answerer factory.

``MEDAPP_ANSWERER`` selects the implementation behind ``/predict`` and
``dev_eval``:

* ``legacy`` (default) — the frozen NLI pipeline (``answerers.legacy``).
* ``modernbert`` — the task-trained candidate scorer (``answerers.modernbert``).

The default stays ``legacy`` until the new answerer beats it out of fold.
"""

import os
from typing import Optional

from .base import Answer, Answerer, Span

DEFAULT = "legacy"
_KNOWN = ("legacy", "modernbert")

__all__ = ["Answer", "Answerer", "Span", "get_answerer", "DEFAULT"]


def get_answerer(name: Optional[str] = None) -> Answerer:
    """Build the answerer selected by ``MEDAPP_ANSWERER`` (or ``name``)."""
    selected = (name or os.environ.get("MEDAPP_ANSWERER", DEFAULT)).strip().lower()

    if selected == "legacy":
        from .legacy import build_answerer

        return build_answerer()

    if selected == "modernbert":
        from .modernbert import build_answerer

        return build_answerer()

    raise ValueError(
        f"Unknown MEDAPP_ANSWERER={selected!r}; expected one of {_KNOWN}."
    )
