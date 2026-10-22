"""Recuperacao: filtro, BM25 e vetorial, fusao RRF, re-ranking e MMR.

Fluxo de `Retriever.retrieve`:
1. filtro por metadado sobre as avaliacoes; viram mascara de entradas do indice (D09);
2. os 50 melhores do BM25 e os 50 melhores da busca vetorial, entre as entradas
   permitidas (D07). Com HyDE, a busca vetorial usa o embedding de uma avaliacao
   hipotetica gerada pelo LLM em vez do embedding da pergunta;
3. Reciprocal Rank Fusion com k = 60: score = soma de 1 / (60 + posicao) nas duas
   listas. Usa so a posicao, entao nao precisa calibrar as escalas do BM25 e do cosseno;
4. re-ranking dos candidatos com cross-encoder, que le pergunta e avaliacao juntas (D08);
5. MMR para nao entregar ao LLM dez variacoes da mesma frase.

As variantes da ablacao (so BM25, so vetorial, sem re-ranking, sem filtro) saem dos
parametros de `retrieve`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.indexing.lexical import LexicalIndex
from src.indexing.vetorial import VectorIndex, embed
from src.retrieval.filtros import Filters, entry_counts, review_mask
from src.retrieval.mmr import mmr

DOCS = Path("data/processed/documentos.parquet")
RRF_K = 60
N_CANDIDATES = 50


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: ordena pela soma de 1 / (k + posicao), posicao a partir de 1."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for pos, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + pos)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


@dataclass
class Hit:
    doc_id: str
    score: float
    text: str
    n_reviews: int
    review_ids: list[str] = field(default_factory=list)
    rank_lexical: int | None = None
    rank_vector: int | None = None
    score_rerank: float | None = None


class Retriever:
    def __init__(self, docs: pd.DataFrame, lexical: LexicalIndex, vector: VectorIndex,
                 reranker=None):
        if lexical.doc_ids != vector.doc_ids:
            raise ValueError("indices lexical e vetorial com entradas em ordem diferente")
        self.docs = docs
        self.lexical = lexical
        self.vector = vector
        self.reranker = reranker
        self.doc_ids = vector.doc_ids
        self.position = {d: i for i, d in enumerate(self.doc_ids)}
        rep = docs[docs["is_representative"]].set_index("doc_id")
        self.texts = rep["text_clean"].reindex(self.doc_ids).tolist()
        self._reviews_by_doc = docs.sort_values("review_creation_date").groupby("doc_id")[
            "review_id"].agg(list)

    @classmethod
    def load(cls, reranker=None) -> Retriever:
        return cls(pd.read_parquet(DOCS), LexicalIndex.load(), VectorIndex.load(), reranker)

    def allowed(self, filters: Filters | None) -> tuple[np.ndarray, pd.Series]:
        """Mascara de entradas e mascara de avaliacoes para um filtro."""
        if filters is None or filters.is_empty():
            mask = pd.Series(True, index=self.docs.index)
        else:
            mask = review_mask(self.docs, filters)
        return entry_counts(self.docs, mask, self.doc_ids), mask

    def candidates(self, question: str, filters: Filters | None = None, mode: str = "hybrid",
                   k: int = N_CANDIDATES, query_vector: np.ndarray | None = None,
                   counts: np.ndarray | None = None) -> list[dict]:
        """Candidatos antes do re-ranking, com as posicoes em cada lista."""
        if counts is None:
            counts, _ = self.allowed(filters)
        allowed = counts > 0
        lex, vec = [], []
        if mode in ("bm25", "hybrid"):
            lex = [d for d, _ in self.lexical.search(question, k=k, allowed=allowed)]
        if mode in ("vector", "hybrid"):
            qv = query_vector if query_vector is not None else embed([question])[0]
            vec = [d for d, _ in self.vector.search(qv, k=k, allowed=allowed)]
        if mode == "bm25":
            fused = [(d, 1.0 / (RRF_K + i)) for i, d in enumerate(lex, 1)]
        elif mode == "vector":
            fused = [(d, 1.0 / (RRF_K + i)) for i, d in enumerate(vec, 1)]
        else:
            fused = rrf([lex, vec])[:k]
        lex_pos = {d: i for i, d in enumerate(lex, 1)}
        vec_pos = {d: i for i, d in enumerate(vec, 1)}
        return [{"doc_id": d, "score": s, "rank_lexical": lex_pos.get(d),
                 "rank_vector": vec_pos.get(d)} for d, s in fused]

    def retrieve(self, question: str, filters: Filters | None = None, mode: str = "hybrid",
                 k: int = 10, n_candidates: int = N_CANDIDATES, rerank: bool = True,
                 mmr_lambda: float | None = 0.7,
                 query_vector: np.ndarray | None = None) -> list[Hit]:
        counts, mask = self.allowed(filters)
        cands = self.candidates(question, filters, mode, n_candidates, query_vector, counts)
        if not cands:
            return []
        if rerank and self.reranker is not None:
            texts = [self.texts[self.position[c["doc_id"]]] for c in cands]
            for c, s in zip(cands, self.reranker.score(question, texts), strict=True):
                c["score_rerank"] = float(s)
                c["score"] = float(s)
            cands.sort(key=lambda c: -c["score"])
        if mmr_lambda is not None and len(cands) > k:
            idx = [self.position[c["doc_id"]] for c in cands]
            rel = np.array([c["score"] for c in cands], dtype=float)
            order = mmr(rel, self.vector.vectors[idx], k=k, lambda_=mmr_lambda)
            cands = [cands[i] for i in order]
        passing = set(self.docs.loc[mask, "review_id"])
        hits = []
        for c in cands[:k]:
            pos = self.position[c["doc_id"]]
            reviews = [r for r in self._reviews_by_doc[c["doc_id"]] if r in passing]
            hits.append(Hit(doc_id=c["doc_id"], score=c["score"], text=self.texts[pos],
                            n_reviews=int(counts[pos]), review_ids=reviews,
                            rank_lexical=c.get("rank_lexical"),
                            rank_vector=c.get("rank_vector"),
                            score_rerank=c.get("score_rerank")))
        return hits
