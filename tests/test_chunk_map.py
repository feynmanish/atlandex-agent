from eval.chunk_map import build_spans, observations, violations
from eval.import_existing import build_records

CAL = [
    {"videoid": "V", "gold": [1], "l2": [1, 3, 2], "mcp": [[1, 130, 0.3], [2, 700, 0.6], [3, 400, 0.7]]},
    {"videoid": "V", "gold": [3], "l2": [3, 1, 0], "mcp": [[1, 710, 0.3], [2, 150, 0.6], [3, None, 0.7]]},
]


def test_observations_align_rank_with_production_order_and_skip_untimed():
    assert observations(CAL) == [("V", 1, 130), ("V", 1, 150), ("V", 2, 400), ("V", 3, 700), ("V", 3, 710)]


def test_consistent_time_order_has_no_violations():
    assert violations(observations(CAL)) == []


def test_swapped_order_is_reported_as_unreliable():
    bad = [("V", 1, 500), ("V", 2, 100)]  # a higher chunk index starts earlier
    assert violations(bad) == [("V", (1, 500), (2, 100))]


def test_spans_move_the_edge_to_the_midpoint_of_observed_neighbours():
    spans = build_spans(observations(CAL))["V"]
    assert spans[1] == {"start": 130, "end": (150 + 400) // 2, "observations": 2}
    assert spans[2]["start"] == (150 + 400) // 2 and spans[2]["end"] == (400 + 700) // 2
    assert spans[3]["start"] == (400 + 700) // 2 and spans[3]["end"] == 710


QUERIES = {"queries": [{"videoid": "V", "title": "T", "question": "q?", "relevant_chunk_ids": [1, 9]}]}


def test_build_records_maps_observed_chunks_and_leaves_unobserved_ones_out():
    m = {"reliable": True, "spans": {"V": {"1": {"start": 130, "end": 275}}}}
    r = build_records(QUERIES, m)[0]
    assert r["gold_video_ids"] == ["V"] and r["gold_chunks"] == [1, 9]
    assert r["gold_segments"] == [{"video": "V", "start": 130, "end": 275, "chunk": 1}]
    assert "https://www.youtube.com/watch?v=V" in r["question"] and r["question_raw"] == "q?"


def test_build_records_falls_back_to_video_level_when_the_map_is_unreliable():
    r = build_records(QUERIES, {"reliable": False, "spans": {"V": {"1": {"start": 1, "end": 2}}}})[0]
    assert r["gold_segments"] == [] and r["gold_basis"].startswith("video-level only")
