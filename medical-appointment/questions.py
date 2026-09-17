"""Rewrite a yes/no question as a declarative proposition for NLI.

An NLI cross-encoder wants a claim to test, not an interrogative. The supplied
questions are template-generated and mostly fall into two shapes:

* auxiliary-initial — ``Was the prescribed dose 200 mg daily?``
* tag questions — ``The lipid profile came back normal, didn't it?``

This is deliberately rule-based and dependency-free: it runs locally, per
request, with no model. The transformation is approximate on purpose; the
verifier is measured against both the proposition and the raw question
(see ``verifier/nli.py``), so a rule that hurts shows up in the trial log.
"""

import re
from typing import List

_AUX = {
    "is", "are", "was", "were", "am", "do", "does", "did", "have", "has", "had",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
}
_BE_FORMS = {"be", "been", "being", "am", "is", "are", "was", "were"}
# Prepositions that start a predicate/complement. Deliberately excludes
# noun-phrase-internal ones (of, in, on, at) so subjects like
# "the cause of the tiredness" are not cut in half.
_COMPLEMENT_PREPS = {
    "without", "with", "after", "before", "around", "about", "from", "to",
    "under", "over", "into", "due", "alongside", "within", "across",
}
_DETERMINERS = {
    "the", "a", "an", "this", "that", "these", "those", "his", "her", "its",
    "their", "our", "your", "my",
}
_ADVERBS = {"still", "already", "also", "currently", "now", "often", "always",
            "never", "only", "again"}
_NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "half", "both", "several",
}
_VERBS = {
    "need", "needs", "take", "takes", "have", "has", "want", "wants", "report",
    "reports", "mention", "mentions", "discuss", "discusses", "talk", "talks",
    "attend", "attends", "come", "comes", "concern", "concerns", "continue",
    "continues", "begin", "begins", "found", "find", "finds", "taken", "give",
    "given", "prescribe", "prescribed", "understand", "understands", "stay",
    "stays", "treat", "treated", "use", "used", "check", "checked", "show",
    "shows", "say", "says", "tell", "tells", "identify", "identified",
    "attribute", "attributed", "assess", "assessed",
    "renew", "renewed", "collect", "collected", "represent", "involve",
    "involves", "last", "run", "runs", "help", "helps", "receive", "received",
    "see", "seen", "remain", "remained", "start", "started", "refer",
    "referred", "discontinue", "discontinued", "increase", "increased",
    "listen", "listened", "come", "came", "describe", "described", "happen",
    "happened", "deny", "denies", "become", "becomes", "mean", "means",
    "regard", "regards", "develop", "develops", "developed", "thought",
    "think", "thinks", "judge", "judged", "follow", "follows", "followed",
    "mentioned", "reported", "discussed", "talked", "attended", "owns",
    "owning", "receives", "discussing", "taking", "having", "using", "coming",
    "needing", "planning", "staying", "getting", "worsen", "worsened",
    "improve", "improved", "occur", "occurred", "persist", "persisted",
    "rule", "rules", "ruled", "feel", "feels", "felt", "sit", "sits", "sat",
    "work", "works", "worked", "express", "expresses", "look", "looks",
    "looked", "seem", "seems", "seemed", "sound", "sounds", "sounded",
    "appear", "appears", "appeared", "read", "reads", "measure", "measured",
    "belong", "belongs", "cause", "caused", "result", "resulted",
}
_ADJECTIVES = {
    "normal", "abnormal", "stable", "unstable", "unchanged", "well", "better",
    "worse", "clear", "unclear", "ready", "sure", "able", "likely", "unlikely",
    "present", "absent", "positive", "negative", "high", "low", "mild",
    "severe", "irregular", "regular", "allergic", "specific", "nonspecific",
    "preventive", "continued", "suitable", "fine", "unremarkable", "intact",
    "unexplained", "concerned", "significant", "mild", "chronic", "red",
    "swollen", "sore", "tender", "itchy", "stiff", "painful", "free",
}
_PARTICIPLE_ENDINGS = ("ed", "ing", "en", "wn")

_TAG_RE = re.compile(
    r",\s*(?:"
    r"right|correct|yes|no|true|"
    r"isn'?t\s+it|aren'?t\s+they|wasn'?t\s+it|weren'?t\s+they|"
    r"doesn'?t\s+it|don'?t\s+they|didn'?t\s+(?:it|they)|"
    r"hasn'?t\s+it|haven'?t\s+they|won'?t\s+it|wouldn'?t\s+it|"
    r"can'?t\s+it|couldn'?t\s+it|is\s+it\s+not"
    r")\s*\??$",
    re.IGNORECASE,
)


def _clean(token: str) -> str:
    return re.sub(r"[^a-z0-9']", "", token.lower())


def _is_number(token: str) -> bool:
    cleaned = _clean(token)
    if not cleaned:
        return False
    if cleaned.isdigit():
        return True
    try:
        float(cleaned)
        return True
    except ValueError:
        pass
    return cleaned in _NUMBER_WORDS


def _starts_complement(token: str) -> bool:
    """Does this token begin the predicate/complement of the subject?"""
    cleaned = _clean(token)
    if not cleaned:
        return False
    if _is_number(token):
        return True
    if cleaned in _COMPLEMENT_PREPS or cleaned in _BE_FORMS or cleaned in _AUX:
        return True
    if cleaned in _VERBS or cleaned in _ADJECTIVES:
        return True
    if len(cleaned) > 4 and cleaned.endswith(_PARTICIPLE_ENDINGS):
        return True
    return False


def strip_tag(question: str) -> str:
    """Remove a trailing tag question (``..., right?`` / ``..., didn't it?``)."""
    return _TAG_RE.sub("", question.strip()).strip()


def _finish(text: str) -> str:
    text = " ".join(text.split())
    if not text:
        return text
    if not text.endswith((".", "?", "!")):
        text += "."
    return text[0].upper() + text[1:]


def to_proposition(question: str) -> str:
    """Turn a yes/no question into a declarative statement."""
    raw = question.strip()
    stripped = strip_tag(raw)
    had_tag = stripped != raw

    text = stripped
    had_question_mark = text.endswith("?")
    if had_question_mark:
        text = text[:-1].strip()

    tokens = text.split()
    if not tokens:
        return question.strip()

    # Only genuine interrogatives get inverted: a statement ("The patient has
    # asthma.") and a tag-stripped statement are already declarative.
    if not had_question_mark or had_tag:
        return _finish(text)

    aux_index = None
    for i, token in enumerate(tokens[:6]):
        if _clean(token) in _AUX:
            aux_index = i
            break

    if aux_index is None:
        return _finish(text)

    prefix: List[str] = tokens[:aux_index]
    aux = tokens[aux_index].lower()
    rest = tokens[aux_index + 1:]

    if rest and _clean(rest[0]) == "there":
        subject = rest[:1]
        complement = rest[1:]
    else:
        stop = len(rest)
        for j, token in enumerate(rest):
            # A determiner after the first token starts a noun-phrase
            # complement ("dry skin | the reason for this consultation").
            if j > 0 and _clean(token) in _DETERMINERS:
                stop = j
                break
            # "to" only starts a complement when it introduces an infinitive
            # ("to be taken"); "admission to hospital" keeps it in the subject.
            if _clean(token) == "to":
                nxt = _clean(rest[j + 1]) if j + 1 < len(rest) else ""
                if nxt in _VERBS or nxt in _BE_FORMS or nxt in _AUX:
                    stop = j
                    break
                continue
            # A word right after a determiner modifies the noun phrase, not the
            # predicate ("the prescribed dose", "a moisturizing cream").
            if j > 0 and _clean(rest[j - 1]) in _DETERMINERS:
                continue
            if _starts_complement(token):
                stop = j
                break
        # Copula with no clear predicate boundary: assume the trailing word is
        # the complement ("the skin intact" -> "the skin" + "intact").
        if stop == len(rest) and aux in _BE_FORMS and len(rest) >= 2:
            stop = len(rest) - 1
        subject = rest[:stop]
        complement = rest[stop:]

    # "the patient still is being treated" -> "the patient is still being treated"
    if aux in _BE_FORMS and len(subject) > 1 and _clean(subject[-1]) in _ADVERBS:
        subject, complement = subject[:-1], [subject[-1]] + complement

    if not subject:
        return _finish(text)

    return _finish(" ".join(prefix + subject + [aux] + complement))
