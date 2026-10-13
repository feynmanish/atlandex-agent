"""The verification agent: what it is shown, how its reply is read, and what happens to the sentences it flags."""
import asyncio
import re
from pathlib import Path

import pytest

from atlandex_agent import prompts
from atlandex_agent.citations import answer_sentences, drop_sentences
from atlandex_agent.graph import NO_COVERAGE, run_agent
from atlandex_agent.verification import build_verify_input, cited_indices, parse_verify_reply, verify_feedback
from fakes import FakeLLM, VID_A, VID_B, json_reply, tool_turn
from test_multi_agent import (ANSWER, NONE_FLAGGED, ROUTE_SIMPLE, SEARCH, SEG_A, SEG_B, search_a, settings, toolbox)

SEGMENTS = {SEG_A: {"text": "scaling laws predict loss"}, SEG_B: {"text": "data quality matters most"}}
TWO = f"Scaling laws predict loss [{SEG_A}]. Data quality matters most [{SEG_B}]."
ONLY_FIRST = f"Scaling laws predict loss [{SEG_A}]."


def flag(*idx, why="not in the source"):
    return json_reply({"unsupported": list(idx), "reasons": {str(i): why for i in idx}})


def both_videos() -> dict:
    return tool_turn(("search_video", {"video": VID_A, "question": "q"}),
                     ("search_video", {"video": VID_B, "question": "q"}))


def run(answers, verify, **kw):
    llm = FakeLLM({"router": [ROUTE_SIMPLE], "supervisor": [SEARCH, ANSWER], "agent": [both_videos(), "READY"],
                   "answer": answers, "verify": verify})
    return llm, asyncio.run(run_agent("q?", llm, toolbox(), settings(**kw), arch="multi"))


# --- pure parts -------------------------------------------------------------------------------------------

def test_verify_input_shows_cited_text_marks_unretrieved_ids_and_skips_uncited_sentences():
    sentences = ["Intro without a source.", f"Scaling laws predict loss [{SEG_A}].",
                 f"Mystery claim [ZZZZZZZZZZZ@5] [{SEG_B}]."]
    text = build_verify_input(sentences, SEGMENTS)
    assert cited_indices(sentences) == [1, 2]
    assert "Intro without a source" not in text and "S0" not in text
    assert f"S1: Scaling laws predict loss.\n  source [{SEG_A}]: scaling laws predict loss" in text
    assert "S2:" in text and "source [ZZZZZZZZZZZ@5]: (never retrieved)" in text
    assert f"source [{SEG_B}]: data quality matters most" in text


def test_parse_verify_reply_reads_fences_think_tags_and_prose():
    valid = {1, 2, 3}
    assert parse_verify_reply('```json\n{"unsupported": [2]}\n```', valid) == {2: ""}
    assert parse_verify_reply('<think>S1 looks {odd}</think> ok {"unsupported": [1, 3], '
                              '"reasons": {"3": "no number in source"}} done', valid) == {1: "", 3: "no number in source"}
    assert parse_verify_reply('{"unsupported": []}', valid) == {}


def test_parse_verify_reply_ignores_out_of_range_and_duplicate_indices():
    assert parse_verify_reply('{"unsupported": [3, 3, 7, -1, "2", true, "x", null]}', {2, 3}) == {2: "", 3: ""}


@pytest.mark.parametrize("text", ["", "   ", "no json here", '{"verdicts": []}', '{"unsupported": "all"}',
                                  '{"unsupported": [1', "[1, 2]"])
def test_an_empty_or_unusable_reply_is_none_never_an_empty_list(text):
    assert parse_verify_reply(text, {1, 2}) is None


def test_feedback_names_the_flagged_sentences_and_their_reasons():
    fb = verify_feedback([("Data quality matters most [X].", "not in source"), ("Other.", "")])
    assert "not supported by the segments they cite" in fb and "Data quality matters most" in fb
    assert "(not in source)" in fb and "or delete them" in fb


def test_drop_sentences_keeps_lines_and_leaves_other_sentences_untouched():
    ans = f"- First [{SEG_A}].\n- Second [{SEG_B}]. Third one [{SEG_A}].\n\n- Fourth [{SEG_B}]."
    assert answer_sentences(ans)[2] == f"Third one [{SEG_A}]."
    assert drop_sentences(ans, [1]) == f"- First [{SEG_A}].\nThird one [{SEG_A}].\n- Fourth [{SEG_B}]."
    assert drop_sentences(ans, []) == "\n".join(line for line in ans.splitlines() if line.strip())
    assert drop_sentences(ans, [0, 1, 2, 3]) == ""


def test_the_verifier_prompt_is_written_apart_from_the_judge_prompt():
    judge = (Path(__file__).parent.parent / "eval" / "judge_prompt.md").read_text(encoding="utf-8")
    norm = lambda t: {re.sub(r"\s+", " ", l).strip() for l in t.splitlines()}  # noqa: E731
    shared = [l for l in norm(prompts.VERIFY_SYSTEM) & norm(judge) if len(l) > 40]
    assert shared == []


# --- in the graph -------------------------------------------------------------------------------------------

def test_nothing_flagged_keeps_the_answer_and_costs_one_verify_call():
    llm, res = run([TWO], [NONE_FLAGGED])
    assert res["answer"] == TWO and llm.roles().count("verify") == 1 and llm.roles().count("answer") == 1
    assert res["verify"]["rounds"] == 1 and res["verify"]["flagged_first"] == []


def test_flagged_first_draft_is_rewritten_checked_and_verified_again_then_dropped_if_flagged_again():
    llm, res = run([TWO, TWO], [flag(1), flag(1)])
    assert llm.roles().count("answer") == 2 and llm.roles().count("verify") == 2
    rewrite = [m for r, m in llm.prompts if r == "answer"][1][1]["content"]
    assert "not supported by the segments they cite" in rewrite and "Data quality matters most" in rewrite
    assert res["answer"] == ONLY_FIRST
    v = res["verify"]
    assert v["rounds"] == 2 and len(v["flagged_first"]) == 1 and len(v["flagged_second"]) == 1
    assert v["dropped_unverified"] == [f"Data quality matters most [{SEG_B}]."]
    assert res["citation_check"]["first_failed"] is False  # the rewrite came from the verifier, not the citation check


def test_a_rewrite_that_fixes_the_flagged_sentence_is_kept():
    llm, res = run([TWO, f"{ONLY_FIRST} Data quality matters [{SEG_B}]."], [flag(1), NONE_FLAGGED])
    assert res["answer"].endswith("Data quality matters [" + SEG_B + "].") and res["verify"]["dropped_unverified"] == []
    assert llm.roles().count("answer") == 2


def test_after_a_citation_rewrite_the_verifier_drops_instead_of_asking_for_a_third_draft():
    first = f"Scaling laws predict loss [{SEG_A}]. Invented [ZZZZZZZZZZZ@5]."  # fails the citation check
    llm, res = run([first, TWO], [flag(1)])
    assert llm.roles().count("answer") == 2  # never a third draft
    assert res["citation_check"]["retried"] and res["verify"]["rounds"] == 1
    assert res["answer"] == ONLY_FIRST and len(res["verify"]["dropped_unverified"]) == 1


def test_a_verifier_that_fails_drops_nothing_and_says_so():
    for broken in ("", "I think sentence 2 is wrong", '{"unsupported": [1'):
        llm, res = run([TWO], [broken])
        assert res["answer"] == TWO and res["verify"]["failed"] and res["verify"]["dropped_unverified"] == []
        assert llm.roles().count("answer") == 1 and any("nothing dropped" in n for n in res["notes"])
        assert res["verify"]["calls"][0]["parsed"] is False


def test_the_verify_calls_are_recorded_with_what_the_smoke_run_needs_to_judge_the_cap():
    _, res = run([TWO], [{"text": "", "finish_reason": "length"}])
    assert res["verify"]["calls"] == [{"completion_tokens": 10, "finish_reason": "length", "empty": True,
                                       "parsed": False}]


def test_dropping_every_sentence_leaves_the_no_coverage_reply():
    _, res = run([ONLY_FIRST, ONLY_FIRST], [flag(0), flag(0)])
    assert res["answer"] == NO_COVERAGE and len(res["verify"]["dropped_unverified"]) == 1


def test_an_index_the_verifier_was_never_shown_is_ignored():
    answer = f"The segments do not cover pricing. Scaling laws predict loss [{SEG_A}]."
    _, res = run([answer], [flag(0)])  # sentence 0 cites nothing, so it was not sent
    assert res["answer"] == answer and res["verify"]["failed"] is False and res["verify"]["flagged_first"] == []


def test_a_verifier_that_flags_everything_in_the_first_round_only_costs_one_rewrite():
    llm, res = run([TWO, TWO, TWO], [flag(0, 1)])
    assert llm.roles().count("answer") == 2 and res["answer"] == NO_COVERAGE


def test_the_verifier_is_shown_each_video_title_and_told_that_naming_the_speaker_is_fine():
    """Smoke finding: the verifier saw only transcript text, so it flagged "<Speaker> says ..." because the speaker's
    name is in the video's metadata and not in the passage. It now sees the title and is told this is not a defect."""
    segs = {SEG_A: {"text": "a good ballpark is a 33% split", "title": "Reiner Pope on scaling"}}
    text = build_verify_input([f"Reiner Pope says a 33% split is a good ballpark [{SEG_A}]."], segs)
    assert f'source [{SEG_A}] ("Reiner Pope on scaling"): a good ballpark is a 33% split' in text
    assert "naming the speaker" in prompts.VERIFY_SYSTEM and "headed by the title" in prompts.VERIFY_SYSTEM
    # a segment without a title (older records, chapter pointers) is shown as before
    assert f"source [{SEG_A}]: a good ballpark" in build_verify_input([f"x [{SEG_A}]."], {SEG_A: {"text": "a good ballpark"}})
