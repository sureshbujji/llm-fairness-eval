"""Unit tests for the fairness harness. All deterministic — no API calls."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from counterfactuals import GROUPS, generate_variants, load_cases
from metrics import (
    counterfactual_consistency,
    demographic_parity_gap,
    evaluate,
    extract_outcome,
    has_stereotype_term,
    is_positive,
    sentiment_gap,
    sentiment_score,
    stereotype_flag_rate,
)
from models import MockBackend


def _result(group, output, case_id="hire-001", scenario="hiring"):
    return {"case_id": case_id, "scenario": scenario, "group": group,
            "name": "Test Name", "output": output,
            "outcome": extract_outcome(output)}


def test_extract_outcome_finds_token():
    assert extract_outcome("YES Strong candidate.") == "YES"
    assert extract_outcome("final rating: needs_improvement") == "NEEDS_IMPROVEMENT"
    assert extract_outcome("nothing relevant in this sentence") is None


def test_is_positive():
    assert is_positive("YES") is True
    assert is_positive("NO") is False
    assert is_positive(None) is None


def test_generate_variants_cover_all_groups():
    case = load_cases()[0]
    variants = generate_variants(case)
    groups = {v["group"] for v in variants}
    assert groups == set(GROUPS)
    assert all("{name}" not in v["prompt"] for v in variants)
    assert all(v["name"] in v["prompt"] for v in variants)


def test_mock_backend_is_biased_on_hiring_only():
    from metrics import extract_outcome
    backend = MockBackend()
    outs = {}
    for case in load_cases():
        for v in generate_variants(case):
            outs.setdefault(v["case_id"], set()).add(
                extract_outcome(backend.complete(v)))
    # hiring case varies by group; everything else is uniform
    assert len(outs["hire-001"]) > 1
    for cid in ("review-001", "loan-001", "salary-001", "triage-001"):
        assert len(outs[cid]) == 1, cid


def test_consistency_detects_bias():
    biased = ([_result("male_european", "YES good")] * 3
              + [_result("male_african", "NO thin")] * 3)
    c = counterfactual_consistency(biased)
    assert c["consistency_rate"] == 0.0
    fair = [_result("male_european", "YES good")] * 3 + [_result("male_african", "YES good")] * 3
    assert counterfactual_consistency(fair)["consistency_rate"] == 1.0


def test_parity_gap():
    results = ([_result("g1", "YES ok")] * 4
               + [_result("g2", "YES ok")] * 2 + [_result("g2", "NO bad")] * 2)
    p = demographic_parity_gap(results)
    assert p["group_positive_rates"] == {"g1": 1.0, "g2": 0.5}
    assert p["parity_gap"] == 0.5
    assert p["worst_group"] == "g2"


def test_sentiment_gap_and_score():
    assert sentiment_score("strong excellent candidate") > 0
    assert sentiment_score("weak thin background") < 0
    assert sentiment_score("") == 0.0
    results = ([_result("g1", "strong excellent work")] * 2
               + [_result("g2", "weak thin background")] * 2)
    assert sentiment_gap(results)["sentiment_gap"] > 0


def test_stereotype_flags():
    assert has_stereotype_term("She was very articulate") is True
    assert has_stereotype_term("Solid engineering background") is False
    results = [_result("g1", "very articulate speaker"),
               _result("g2", "solid background")]
    s = stereotype_flag_rate(results)
    assert s["flagged_count"] == 1
    assert s["flag_rate"] == 0.5


def test_evaluate_end_to_end_with_mock():
    backend = MockBackend()
    results = []
    for case in load_cases():
        for v in generate_variants(case):
            out = backend.complete(v)
            results.append(_result(v["group"], out, v["case_id"], v["scenario"]))
    m = evaluate(results)
    assert m["n_results"] == len(results) > 0
    # mock is biased on hiring -> parity gap must be visible
    assert m["parity"]["parity_gap"] > 0
    assert m["consistency"]["consistency_rate"] < 1.0
