"""Maximal Marginal Relevance: relevancia com penalidade por redundancia.

A cada passo escolhe o candidato que maximiza
    lambda * relevancia - (1 - lambda) * maior similaridade com os ja escolhidos.
A relevancia e reescalada para [0, 1] dentro da lista, para ficar na mesma escala do
cosseno entre os embeddings.
"""

from __future__ import annotations

import numpy as np


def mmr(relevance: np.ndarray, vectors: np.ndarray, k: int, lambda_: float = 0.7) -> list[int]:
    """Ordem dos k candidatos escolhidos (indices na lista de entrada)."""
    n = len(relevance)
    if n == 0:
        return []
    rel = np.asarray(relevance, dtype=float)
    span = rel.max() - rel.min()
    rel = (rel - rel.min()) / span if span > 0 else np.ones(n)
    sim = vectors @ vectors.T
    chosen = [int(np.argmax(rel))]
    remaining = set(range(n)) - set(chosen)
    while remaining and len(chosen) < k:
        cand = np.array(sorted(remaining))
        redundancy = sim[np.ix_(cand, chosen)].max(axis=1)
        value = lambda_ * rel[cand] - (1 - lambda_) * redundancy
        best = int(cand[np.argmax(value)])
        chosen.append(best)
        remaining.remove(best)
    return chosen
