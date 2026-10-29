"""Metricas de recuperacao, bootstrap e concordancia."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.julgamento import cohen_kappa
from src.evaluation.recuperacao import (
    bootstrap_diff,
    compare,
    evaluate,
    objective_precision,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RANKED = ["a", "b", "c", "d", "e"]


def test_precision_counts_relevant_in_top_k():
    assert precision_at_k(RANKED, {"b", "d", "z"}, 2) == 0.5
    assert precision_at_k(RANKED, {"b", "d", "z"}, 5) == 0.4
    # menos resultados que k: o que falta conta como nao relevante
    assert precision_at_k(["a"], {"a"}, 10) == 0.1


def test_recall_uses_all_known_relevant():
    assert recall_at_k(RANKED, {"b", "d", "z"}, 5) == pytest.approx(2 / 3)
    assert np.isnan(recall_at_k(RANKED, set(), 5))


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, {"z"}) == 0.0
    assert reciprocal_rank(RANKED, {"e"}, k=3) == 0.0


def test_objective_precision_averages_share_in_group():
    share = pd.Series({"a": 1.0, "b": 0.5, "c": 0.0})
    assert objective_precision(["a", "b", "c", "x"], share, 4) == pytest.approx(1.5 / 4)


def test_evaluate_builds_one_row_per_question_and_variant():
    runs = pd.DataFrame({"question_id": ["q1"] * 4, "variant": ["v1", "v1", "v2", "v2"],
                         "rank": [1, 2, 1, 2], "doc_id": ["a", "b", "b", "a"]})
    rel = pd.DataFrame({"question_id": ["q1", "q1"], "doc_id": ["b", "c"],
                        "relevante": [True, True]})
    m = evaluate(runs, rel, k=2).set_index("variant")
    assert m.loc["v1", "mrr"] == 0.5 and m.loc["v2", "mrr"] == 1.0
    assert m.loc["v1", "recall@2"] == 0.5 and m.loc["v1", "precision@2"] == 0.5


def test_bootstrap_detects_consistent_difference():
    rng = np.random.default_rng(0)
    b = rng.uniform(0, 0.5, 40)
    mean, lo, hi = bootstrap_diff(b + 0.2, b)
    assert mean == pytest.approx(0.2) and lo > 0
    mean, lo, hi = bootstrap_diff(b, b)
    assert mean == 0 and lo <= 0 <= hi


def test_compare_flags_significance():
    m = pd.DataFrame({"question_id": ["q1", "q2", "q3", "q1", "q2", "q3"],
                      "variant": ["a"] * 3 + ["b"] * 3,
                      "mrr": [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]})
    out = compare(m, "a", "b", "mrr")
    assert out["diferenca"] == pytest.approx(0.5) and out["significativa"]


def test_cohen_kappa():
    a = np.array([0, 1, 2, 2, 0, 1])
    assert cohen_kappa(a, a) == 1.0
    assert cohen_kappa(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])) < 0
