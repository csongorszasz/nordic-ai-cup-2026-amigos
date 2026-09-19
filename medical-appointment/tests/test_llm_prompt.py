"""Prompt builders for L0/L1/L2 (no model)."""

from answerers.llm_prompt import (
    build_few_shot,
    build_l0_messages,
    build_l1_messages,
    build_l2_cite_messages,
    build_l2_decide_messages,
    qid_for,
    serialize_transcript,
    render_example,
    system_prompt,
)


def make_transcript(tid):
    segments = [
        {"id": 0, "start": 0.0, "end": 1.4, "text": f"{tid} one"},
        {"id": 1, "start": 1.5, "end": 2.9, "text": f"{tid} dose is 100 mg"},
        {"id": 2, "start": 3.0, "end": 4.0, "text": f"{tid} three"},
    ]
    words = [
        {"word": " " + token, "start": 0.1 * i, "end": 0.1 + 0.1 * i}
        for i, token in enumerate("a b c d e f".split())
    ]
    return {"segments": segments, "words": words}


def test_qid_and_serialize():
    assert qid_for(0) == "q01"
    assert qid_for(9) == "q10"
    transcript = make_transcript("x")
    text = serialize_transcript(transcript)
    assert "[s00 0.00-1.40] x one" in text
    assert text.count("\n") == 2


def test_l0_messages_contain_transcript_and_questions():
    messages = build_l0_messages(make_transcript("x"), ["Q one?", "Q two?"])
    assert messages[0]["role"] == "system"
    assert "TRANSCRIPT" in messages[1]["content"]
    assert "q01: Q one?" in messages[1]["content"]
    assert "q02: Q two?" in messages[1]["content"]


def test_l1_inserts_few_shot_turns():
    few_shot = [("U1", "A1"), ("U2", "A2")]
    messages = build_l1_messages(make_transcript("x"), ["Q?"], few_shot)
    assert len(messages) == 1 + 2 * 2 + 1
    assert messages[1] == {"role": "user", "content": "U1"}
    assert messages[2] == {"role": "assistant", "content": "A1"}


def test_l2_builders():
    decide = build_l2_decide_messages(make_transcript("x"), ["Q?"])
    assert '"answer":"yes"' in decide[-1]["content"]
    cite = build_l2_cite_messages(make_transcript("x"), [("q01", "Q?")])
    assert "already been answered YES" in cite[-1]["content"]
    assert "q01: Q?" in cite[-1]["content"]


def test_full_context_changes_only_positive_demonstration_visibility():
    transcript = {
        "segments": [
            {"id": index, "start": float(index), "end": index + 0.8, "text": f"line {index}"}
            for index in range(15)
        ],
        "words": [],
    }
    base, target = render_example(transcript, "Q?", True, "line 12", (12.0, 12.8), "base")
    full, full_target = render_example(
        transcript, "Q?", True, "line 12", (12.0, 12.8), "full_context"
    )
    assert "[s00 " not in base and "[s00 " in full
    assert "[s14 " in full
    assert full_target == target
    negative, _ = render_example(transcript, "Q?", False, None, (12.0, 12.8), "full_context")
    assert "[s00 " not in negative


def test_timestamp_ablation_preserves_segment_ids_text_and_targets():
    transcript = make_transcript("x")
    messages = build_l1_messages(transcript, ["Was the dose 100 mg?"], variant="no_timestamps")
    assert "[s01] x dose is 100 mg" in messages[-1]["content"]
    assert "1.50-2.90" not in messages[-1]["content"]
    assert "timestamped transcript" not in messages[0]["content"]
    base_user, base_answer = render_example(transcript, "Q?", True, "dose", (1.5, 2.9), "base")
    user, answer = render_example(transcript, "Q?", True, "dose", (1.5, 2.9), "no_timestamps")
    assert "[s01]" in user and "1.50-2.90" not in user
    assert base_answer == answer


def test_final_statement_rule_preserves_the_actual_question_and_schema():
    transcript = make_transcript("x")
    questions = ["Was the original dose 100 mg?"]
    base = build_l1_messages(transcript, questions, variant="base")
    changed = build_l1_messages(transcript, questions, variant="final_statement")
    assert changed[-1] == base[-1]
    assert "SAME queried fact" in system_prompt("final_statement")
    assert "temporal status" in changed[0]["content"]


def test_build_few_shot_balanced_and_loco_safe():
    rows_by_tid = {
        "s1": [{
            "question_id": "q_s1", "transcript_id": "s1",
            "question_type": "positive", "question": "Was the dose 100 mg?",
            "evidence_start": "1.5", "evidence_end": "2.9",
        }],
        "s2": [{
            "question_id": "q_s2", "transcript_id": "s2",
            "question_type": "hard_negative", "question": "Was the dose 200 mg?",
            "evidence_start": "", "evidence_end": "",
        }],
        "s3": [{
            "question_id": "q_s3", "transcript_id": "s3",
            "question_type": "off_topic", "question": "Did they cook?",
            "evidence_start": "", "evidence_end": "",
        }],
    }
    transcripts = {tid: make_transcript(tid) for tid in rows_by_tid}
    evidence = {"q_s2": {"bucket": "refute", "start": "1.5", "end": "2.9"}}

    few_shot = build_few_shot(rows_by_tid, transcripts, evidence, exclude_tid="s0")
    assert len(few_shot) == 3
    assistants = [assistant for _, assistant in few_shot]
    assert any('"answer": "yes"' in a for a in assistants)
    assert sum('"answer": "no"' in a for a in assistants) == 2

    # Excluding s1 drops the only positive example -> 2 turns remain.
    fewer = build_few_shot(rows_by_tid, transcripts, evidence, exclude_tid="s1")
    assert len(fewer) == 2

    rows_by_tid["s4"] = [{**rows_by_tid["s1"][0], "question_id": "q_s4", "transcript_id": "s4"}]
    transcripts["s4"] = make_transcript("s4")
    more = build_few_shot(
        rows_by_tid, transcripts, evidence, exclude_tid="s0", variant="two_positive"
    )
    assert len(more) == 4
    assert sum('"answer": "yes"' in assistant for _, assistant in more) == 2
    assert sum('"answer": "no"' in assistant for _, assistant in more) == 2


from answerers.llm_prompt import (
    build_rag_messages,
    candidate_ids,
    rag_candidates_block,
)


class _Passage:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


def test_rag_candidates_block_and_messages():
    candidates = [
        [_Passage(100.0, 105.0, "the dose is 100 mg")],
        [_Passage(20.0, 23.0, "no side effects")],
    ]
    questions = ["Was the dose 100 mg?", "Any side effects?"]
    block = rag_candidates_block(questions, candidates)
    assert "q01: Was the dose 100 mg?" in block
    assert "c01 [100.00-105.00] the dose is 100 mg" in block
    # Ids are globally unique, not reset per question.
    assert "c02 [20.00-23.00] no side effects" in block
    assert candidate_ids(candidates) == {"c01": (0, 0), "c02": (1, 0)}

    messages = build_rag_messages(questions, candidates)
    user = messages[-1]["content"]
    assert "c02 [20.00-23.00] no side effects" in user
    assert '"candidate":"c01"' in user
    assert messages[0]["role"] == "system"
