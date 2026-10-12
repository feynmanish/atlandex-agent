"""Fix 4: every factual sentence must cite a segment. Offline; the sentences are shapes seen in real answers."""
import pytest

from atlandex_agent.citations import drop_uncited, is_uncited_claim, uncited_claims
from atlandex_agent.graph import NO_COVERAGE
from atlandex_agent.segments import split_sentences
from fakes import FakeLLM, VID_A, tool_turn
from test_graph import ROUTE_SIMPLE, SEG_A, run, toolbox

CLAIMS = [
    "Vapnik expresses doubt that machines can automatically discover good predicates.",
    "Thus, according to Vapnik, machines cannot reliably discover good predicates on their own.",
    "The video discusses Pope Leo XIII's 1891 encyclical about the industrial revolution's impact on workers.",
    "- **NYFGCESmikA@0** – *DHH: Future of Programming, AI, Agentic Engineering, Vibe Coding & Linux*",
    "Yes, according to Vladimir Vapnik, convolution functions as a **single translational-invariance predicate**.",
]

NOT_CLAIMS = [
    f"Vapnik expresses doubt that machines can discover good predicates [{SEG_A}].",  # it cites
    "The provided segments do not contain information about Amazon lobbying the Vatican.",
    "None of the listed segments explicitly mention these individuals or their perspectives.",
    "The Reiner Pope video does not discuss scaling laws in the provided segments.",
    "However, none of the provided segments contain DHH's explanation for this.",
    "I cannot answer how the two talks describe the outlook because the segments do not say.",
    "No indexed video covered this question well enough to answer it.",
    "**Indexed videos featuring DHH on AI and programming:**",
    "[Segments cited: Burge5lagdM@507, PU_U_Feb369c@1410, Burge5lagdM@583]",
    "Yes.",
    "Frankenstein theory.",
]


@pytest.mark.parametrize("s", CLAIMS)
def test_factual_sentences_without_an_id_are_uncited_claims(s):
    assert is_uncited_claim(s)


@pytest.mark.parametrize("s", NOT_CLAIMS)
def test_citing_sentences_labels_fragments_and_not_covered_statements_are_not_claims(s):
    assert not is_uncited_claim(s)


def test_an_abbreviation_does_not_end_a_sentence():
    text = "The speaker describes Dr. Frankenstein theory. He disagrees with it."
    assert split_sentences(text) == ["The speaker describes Dr. Frankenstein theory.", "He disagrees with it."]
    assert split_sentences('He calls it the "Dr. Frankenstein theory".') == ['He calls it the "Dr. Frankenstein theory".']
    assert uncited_claims("Mr. Smith explains the idea. Dr. Jones disagrees with the idea.") == [
        "Mr. Smith explains the idea.", "Dr. Jones disagrees with the idea."]


def test_drop_uncited_keeps_cited_sentences_not_covered_statements_and_lines():
    answer = (f"Scaling laws predict loss [{SEG_A}]. Thus, bigger is always better.\n"
              "The segments do not cover cost.\n"
              "Vapnik expresses doubt about this idea.")
    cleaned, dropped = drop_uncited(answer)
    assert cleaned == f"Scaling laws predict loss [{SEG_A}].\nThe segments do not cover cost."
    assert dropped == ["Thus, bigger is always better.", "Vapnik expresses doubt about this idea."]


def answer_run(answers):
    llm = FakeLLM({
        "router": [ROUTE_SIMPLE],
        "agent": [tool_turn(("search_video", {"video": VID_A, "question": "q"})), "READY"],
        "answer": answers,
    })
    return llm, run(llm, toolbox())


def test_uncited_claim_triggers_one_retry_and_the_feedback_names_the_sentence():
    llm, res = answer_run([f"Scaling laws predict loss [{SEG_A}]. Thus bigger models always win.",
                           f"Scaling laws predict loss [{SEG_A}]."])
    cc = res["citation_check"]
    assert cc["retried"] and cc["uncited_first"] == ["Thus bigger models always win."]
    assert llm.roles().count("answer") == 2
    retry_prompt = [m for r, m in llm.prompts if r == "answer"][1][1]["content"]
    assert "Thus bigger models always win." in retry_prompt and "end without a segment ID" in retry_prompt
    assert "dropped_uncited" not in cc and res["answer"] == f"Scaling laws predict loss [{SEG_A}]."


def test_claims_still_uncited_after_the_retry_are_dropped_and_counted():
    llm, res = answer_run([f"Scaling laws predict loss [{SEG_A}]. Thus bigger models always win.",
                           f"Scaling laws predict loss [{SEG_A}]. Thus bigger models always win. Also more compute reliably helps."])
    cc = res["citation_check"]
    assert llm.roles().count("answer") == 2  # one retry, never a third
    assert cc["dropped_uncited"] == ["Thus bigger models always win.", "Also more compute reliably helps."]
    assert res["answer"] == f"Scaling laws predict loss [{SEG_A}]."
    assert res["cited_ids"] == [SEG_A]


def test_an_answer_that_cites_nothing_is_dropped_to_no_coverage():
    _, res = answer_run(["Scaling laws predict loss.", "Scaling laws still predict loss."])
    assert res["answer"] == NO_COVERAGE
    assert res["citation_check"]["dropped_uncited"] == ["Scaling laws still predict loss."]


def test_not_covered_statements_need_no_citation_and_cause_no_retry():
    llm, res = answer_run([f"Scaling laws predict loss [{SEG_A}]. The segments do not cover model cost."])
    assert llm.roles().count("answer") == 1 and not res["citation_check"]["retried"]
    assert "do not cover" in res["answer"]


def test_unretrieved_ids_and_uncited_claims_are_fixed_in_one_retry():
    llm, res = answer_run([f"Made up [ZZZZZZZZZZZ@5]. Thus bigger models always win.",
                           f"Scaling laws predict loss [{SEG_A}]."])
    cc = res["citation_check"]
    assert cc["unsupported_first"] == ["ZZZZZZZZZZZ@5"] and cc["uncited_first"] == ["Thus bigger models always win."]
    assert llm.roles().count("answer") == 2 and res["cited_ids"] == [SEG_A]
