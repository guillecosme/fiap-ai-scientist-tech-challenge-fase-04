"""Gate de evidencia e calibracao do limiar."""

import numpy as np

from src.graph.gate import Gate, calibrate, save_threshold, threshold_curve

HITS = [{"score": 0.1, "score_rerank": 2.0}, {"score": 0.1, "score_rerank": -1.0}]


def test_passes_when_top_score_reaches_threshold():
    d = Gate(threshold=1.5).decide(HITS, total_recorte=50)
    assert d["passa"] and d["top_score"] == 2.0 and d["acima_do_limiar"] == 1


def test_refuses_weak_evidence():
    d = Gate(threshold=3.0).decide(HITS, total_recorte=50)
    assert not d["passa"] and d["motivo"] == "evidencia_insuficiente"


def test_refuses_empty_or_tiny_scope():
    assert Gate().decide([], 0)["motivo"] == "sem_avaliacoes"
    assert Gate(threshold=-9, min_reviews=3).decide(HITS, 2)["motivo"] == "poucas_avaliacoes"


def test_uses_fusion_score_when_there_is_no_rerank():
    d = Gate(threshold=0.05).decide([{"score": 0.1, "score_rerank": None}], 10)
    assert d["passa"]


def test_calibration_separates_the_two_groups():
    scores = np.array([3.0, 2.5, 2.0, -1.0, -2.0, np.nan])
    answerable = np.array([True, True, True, False, False, False])
    t, curve = calibrate(scores, answerable)
    assert -1.0 < t < 2.0
    best = curve.loc[curve["limiar"] == t].iloc[0]
    assert best["acuracia_balanceada"] == 1.0


def test_calibration_prefers_lower_threshold_on_ties():
    scores = np.array([1.0, 0.0])
    t, _ = calibrate(scores, np.array([True, False]))
    assert t == 0.5


def test_curve_counts_missing_results_as_refusal():
    curve = threshold_curve(np.array([np.nan, 1.0]), np.array([True, False]))
    assert curve["passa_respondiveis"].max() == 0.0 or np.isinf(curve["limiar"]).any()


def test_save_and_load(tmp_path):
    path = tmp_path / "limiar.json"
    save_threshold(1.25, path, perguntas=40)
    assert Gate.load(path).threshold == 1.25
