"""Indice vetorial: embeddings multilingues dos textos distintos.

Uso:
    python -m src.indexing.vetorial    # calcula os embeddings e grava data/index/embeddings.npz

Modelo: paraphrase-multilingual-MiniLM-L12-v2 (D05), o mesmo das aulas. E um
bi-encoder treinado para parafrases em mais de 50 linguas, com saida de 384 dimensoes;
os vetores sao normalizados, entao o produto interno e o cosseno.

O texto que vai ao modelo e o texto limpo (D03), uma vez por doc_id (D04). Os vetores
ficam em disco em float16, metade do tamanho, e voltam para float32 na carga.

O indice e um FAISS IndexFlatIP (D06): busca exata por produto interno. Com cerca de
35 mil vetores a busca exata leva milissegundos, e um indice aproximado (HNSW, IVF)
so trocaria precisao por uma velocidade que nao faz falta nesta escala. O indice e
montado na carga a partir dos vetores; os filtros por metadado chegam como mascara de
entradas permitidas e viram um seletor de ids do FAISS.
"""

from __future__ import annotations

import json
import os
from functools import cache
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from src.preprocessing.deduplicacao import unique_docs

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DOCS = Path("data/processed/documentos.parquet")
INDEX_DIR = Path("data/index")
EMBEDDINGS = INDEX_DIR / "embeddings.npz"
BATCH_SIZE = 64


@cache
def load_model():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed(texts: list[str], batch_size: int = BATCH_SIZE,
          show_progress: bool = False) -> np.ndarray:
    """Vetores normalizados (norma 1) em float32."""
    vectors = load_model().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=show_progress)
    return vectors.astype(np.float32)


def build_embeddings(docs: pd.DataFrame,
                     show_progress: bool = True) -> tuple[list[str], np.ndarray]:
    uniq = unique_docs(docs)
    vectors = embed(uniq["text_clean"].tolist(), show_progress=show_progress)
    return uniq["doc_id"].tolist(), vectors


def save_embeddings(doc_ids: list[str], vectors: np.ndarray, path: Path = EMBEDDINGS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, doc_ids=np.array(doc_ids), vectors=vectors.astype(np.float16))


def load_embeddings(path: Path = EMBEDDINGS) -> tuple[list[str], np.ndarray]:
    data = np.load(path, allow_pickle=False)
    vectors = data["vectors"].astype(np.float32)
    # a conversao para float16 tira a norma de 1 por pouco; renormaliza
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return data["doc_ids"].tolist(), vectors


class VectorIndex:
    def __init__(self, doc_ids: list[str], vectors: np.ndarray):
        self.doc_ids = list(doc_ids)
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)

    @classmethod
    def load(cls, path: Path = EMBEDDINGS) -> VectorIndex:
        return cls(*load_embeddings(path))

    def __len__(self) -> int:
        return len(self.doc_ids)

    def scores(self, query: str | np.ndarray) -> np.ndarray:
        """Cosseno da pergunta com todas as entradas, na ordem de doc_ids."""
        q = embed([query])[0] if isinstance(query, str) else query
        return self.vectors @ q

    def search(self, query: str | np.ndarray, k: int = 10,
               allowed: np.ndarray | None = None) -> list[tuple[str, float]]:
        q = embed([query]) if isinstance(query, str) else query.reshape(1, -1)
        params = None
        if allowed is not None:
            ids = np.flatnonzero(allowed).astype("int64")
            if len(ids) == 0:
                return []
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(ids))
            k = min(k, len(ids))
        dist, idx = self.index.search(q.astype(np.float32), min(k, len(self)), params=params)
        return [(self.doc_ids[i], float(d)) for d, i in zip(dist[0], idx[0], strict=True)
                if i >= 0]


def main() -> None:
    doc_ids, vectors = build_embeddings(pd.read_parquet(DOCS))
    save_embeddings(doc_ids, vectors)
    meta = {"model": MODEL_NAME, "dim": int(vectors.shape[1]), "n_docs": len(doc_ids),
            "normalized": True, "dtype_on_disk": "float16", "faiss_index": "IndexFlatIP",
            "text_field": "text_clean", "doc_order": "unique_docs (doc_id ascendente)"}
    (INDEX_DIR / "embeddings.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{len(doc_ids):,} embeddings de {vectors.shape[1]} dimensoes -> {EMBEDDINGS}")


if __name__ == "__main__":
    main()
