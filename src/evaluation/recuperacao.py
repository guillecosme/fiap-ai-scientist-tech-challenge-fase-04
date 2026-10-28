"""Metricas de recuperacao e comparacao entre variantes.

Definicoes, por pergunta, sobre a lista ordenada de entradas devolvida pela variante:
- precision@k: entradas relevantes entre as k primeiras, dividido por k;
- recall@k: entradas relevantes entre as k primeiras, dividido pelo total de entradas
  relevantes conhecidas para a pergunta (as do pool julgadas relevantes). Perguntas sem
  nenhuma entrada relevante ficam fora da media de recall;
- MRR: inverso da posicao da primeira entrada relevante (zero se nenhuma aparece);
- precisao objetiva: fracao das k primeiras cujas avaliacoes pertencem ao grupo definido
  pelo metadado (atraso ou nao recebido), sem juiz.

As diferencas entre variantes vem com intervalo de confianca de 95% por bootstrap
pareado sobre as perguntas: as mesmas perguntas sorteadas para as duas variantes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(d in relevant for d in ranked[:k]) / k


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    return sum(d in relevant for d in ranked[:k]) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str], k: int | None = None) -> float:
    for pos, d in enumerate(ranked[:k] if k else ranked, start=1):
        if d in relevant:
            return 1.0 / pos
    return 0.0


def objective_precision(ranked: list[str], share_in_group: pd.Series, k: int) -> float:
    """Media, nas k primeiras, da fracao de avaliacoes da entrada que estao no grupo."""
    top = ranked[:k]
    if not top:
        return float("nan")
    return float(share_in_group.reindex(top).fillna(0).sum() / k)


def evaluate(runs: pd.DataFrame, relevance: pd.DataFrame, k: int = 10,
             column: str = "relevante") -> pd.DataFrame:
    """Metricas por pergunta e variante.

    runs: question_id, variant, rank, doc_id (saida das variantes).
    relevance: question_id, doc_id e a coluna booleana de relevancia.
    """
    rel = relevance[relevance[column]].groupby("question_id")["doc_id"].agg(set)
    rows = []
    for (qid, variant), g in runs.groupby(["question_id", "variant"], sort=False):
        ranked = g.sort_values("rank")["doc_id"].tolist()
        r = rel.get(qid, set())
        rows.append({"question_id": qid, "variant": variant,
                     f"precision@{k}": precision_at_k(ranked, r, k),
                     f"recall@{k}": recall_at_k(ranked, r, k),
                     "mrr": reciprocal_rank(ranked, r, k),
                     "n_relevantes": len(r), "n_devolvidos": len(ranked)})
    return pd.DataFrame(rows)


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n: int = 5000,
                   seed: int = 42) -> tuple[float, float, float]:
    """Diferenca media (a - b) pareada e intervalo de 95% por bootstrap das perguntas."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    d = a[keep] - b[keep]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(metrics: pd.DataFrame, base: str, other: str, metric: str) -> dict:
    """Diferenca de `metric` entre duas variantes nas mesmas perguntas."""
    wide = metrics.pivot(index="question_id", columns="variant", values=metric)
    mean, lo, hi = bootstrap_diff(wide[other].to_numpy(), wide[base].to_numpy())
    return {"comparacao": f"{other} - {base}", "metrica": metric, "diferenca": mean,
            "ic95_inf": lo, "ic95_sup": hi, "significativa": bool(lo > 0 or hi < 0)}
