"""Indice lexical: BM25 sobre os radicais dos textos distintos.

Uso:
    python -m src.indexing.lexical    # grava data/index/bm25.parquet

O indice tem uma entrada por doc_id (texto distinto depois da normalizacao, D04), na
ordem de `unique_docs`. O que vai para o disco sao os radicais de cada entrada e os
parametros; o BM25 e reconstruido na carga, o que leva uma fracao de segundo e evita
depender de pickle.

A busca aceita uma mascara booleana sobre as entradas (`allowed`), que e como os
filtros por metadado chegam ao indice: so as entradas permitidas concorrem ao top-k.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from src.preprocessing.deduplicacao import unique_docs
from src.preprocessing.tokenizacao import tokenize

DOCS = Path("data/processed/documentos.parquet")
INDEX_DIR = Path("data/index")
BM25_FILE = INDEX_DIR / "bm25.parquet"
BM25_PARAMS = INDEX_DIR / "bm25.json"

K1 = 1.5
B = 0.75


def _top_k(scores: np.ndarray, k: int, allowed: np.ndarray | None) -> list[tuple[int, float]]:
    """Posicoes e scores das k maiores entradas permitidas com score positivo."""
    scores = scores.astype(float).copy()
    if allowed is not None:
        scores[~allowed] = -np.inf
    k = min(k, int(np.isfinite(scores).sum()))
    if k <= 0:
        return []
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]


@dataclass
class LexicalIndex:
    doc_ids: list[str]
    tokens: list[list[str]]
    k1: float = K1
    b: float = B
    bm25: BM25Okapi = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.bm25 = BM25Okapi(self.tokens, k1=self.k1, b=self.b)

    @classmethod
    def from_documents(cls, docs: pd.DataFrame) -> LexicalIndex:
        uniq = unique_docs(docs)
        return cls(uniq["doc_id"].tolist(), [list(t) for t in uniq["tokens"]])

    def scores(self, query: str) -> np.ndarray:
        """Score BM25 da pergunta contra todas as entradas, na ordem de doc_ids."""
        terms = tokenize(query)
        if not terms:
            return np.zeros(len(self.doc_ids))
        return self.bm25.get_scores(terms)

    def search(self, query: str, k: int = 10,
               allowed: np.ndarray | None = None) -> list[tuple[str, float]]:
        return [(self.doc_ids[i], s) for i, s in _top_k(self.scores(query), k, allowed)]

    def save(self, path: Path = BM25_FILE, params_path: Path = BM25_PARAMS) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"doc_id": self.doc_ids, "tokens": self.tokens}).to_parquet(
            path, index=False, compression="zstd")
        params_path.write_text(json.dumps({"k1": self.k1, "b": self.b,
                                           "n_docs": len(self.doc_ids)}, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path = BM25_FILE, params_path: Path = BM25_PARAMS) -> LexicalIndex:
        df = pd.read_parquet(path)
        params = json.loads(params_path.read_text())
        return cls(df["doc_id"].tolist(), [list(t) for t in df["tokens"]],
                   k1=params["k1"], b=params["b"])


def main() -> None:
    index = LexicalIndex.from_documents(pd.read_parquet(DOCS))
    index.save()
    vocab = len({t for toks in index.tokens for t in toks})
    print(f"BM25: {len(index.doc_ids):,} entradas, {vocab:,} radicais -> {BM25_FILE}")


if __name__ == "__main__":
    main()
