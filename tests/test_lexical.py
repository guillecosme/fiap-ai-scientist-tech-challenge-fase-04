"""Indice lexical: BM25, filtro por mascara, persistencia e TF-IDF de comparacao."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.indexing.lexical import LexicalIndex, TfidfIndex
from src.preprocessing.tokenizacao import tokenize

TEXTS = {
    "d1": "produto chegou quebrado e mal embalado",
    "d2": "não recebi o produto ainda",
    "d3": "recebi o produto antes do prazo",
    "d4": "entrega rápida, recomendo",
    "d5": "veio o produto errado, cor diferente",
}


@pytest.fixture(scope="module")
def index() -> LexicalIndex:
    return LexicalIndex(list(TEXTS), [tokenize(t) for t in TEXTS.values()])


def test_query_term_ranks_matching_document_first(index):
    assert index.search("chegou quebrado", k=1)[0][0] == "d1"
    assert index.search("produto errado", k=1)[0][0] == "d5"


def test_negation_changes_the_ranking(index):
    assert index.search("não recebi", k=1)[0][0] == "d2"


def test_accent_and_inflection_insensitive(index):
    assert index.search("entregas rapidas", k=1)[0][0] == "d4"


def test_documents_without_query_terms_are_not_returned(index):
    ids = [d for d, _ in index.search("quebrado", k=5)]
    assert ids == ["d1"]


def test_allowed_mask_restricts_candidates(index):
    allowed = np.array([False, False, True, False, False])
    hits = index.search("produto", k=5, allowed=allowed)
    assert [d for d, _ in hits] == ["d3"]


def test_empty_query_returns_nothing(index):
    assert index.search("o de a", k=3) == []


def test_save_and_load_roundtrip(index, tmp_path: Path):
    index.save(tmp_path / "bm25.parquet", tmp_path / "bm25.json")
    loaded = LexicalIndex.load(tmp_path / "bm25.parquet", tmp_path / "bm25.json")
    assert loaded.doc_ids == index.doc_ids
    np.testing.assert_allclose(loaded.scores("produto quebrado"), index.scores("produto quebrado"))


def test_tfidf_comparison_index(index):
    tfidf = TfidfIndex.from_lexical(index)
    assert tfidf.search("chegou quebrado", k=1)[0][0] == "d1"
    assert len(tfidf.scores("produto")) == len(TEXTS)


INDEX = Path("data/index/bm25.parquet")


@pytest.mark.skipif(not INDEX.exists(), reason="indice ausente")
def test_built_index_matches_documents():
    docs = pd.read_parquet("data/processed/documentos.parquet")
    index = LexicalIndex.load()
    assert len(index.doc_ids) == docs["doc_id"].nunique()
    assert len(set(index.doc_ids)) == len(index.doc_ids)
