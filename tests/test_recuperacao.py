"""Filtros por metadado, regras de extracao, RRF, re-ranking e MMR."""

import numpy as np
import pandas as pd
import pytest

from src.indexing.lexical import LexicalIndex
from src.indexing.vetorial import VectorIndex
from src.preprocessing.tokenizacao import tokenize
from src.retrieval.filtros import Filters, entry_counts, review_mask, rule_based_filters
from src.retrieval.hibrida import Retriever, rrf
from src.retrieval.mmr import mmr


def make_docs() -> pd.DataFrame:
    """Cinco avaliacoes em quatro textos; r4 e r5 compartilham o texto d4."""
    return pd.DataFrame({
        "review_id": ["r1", "r2", "r3", "r4", "r5"],
        "doc_id": ["d1", "d2", "d3", "d4", "d4"],
        "is_representative": [True, True, True, True, False],
        "text_clean": ["produto chegou quebrado", "não recebi o pedido ainda",
                       "chegou antes do prazo, recomendo", "muito bom", "muito bom"],
        "review_score": [1, 1, 5, 5, 3],
        "delivery_status": ["no prazo", "nao entregue", "no prazo", "atrasada", "no prazo"],
        "answered_before_delivery": [False, True, False, False, False],
        "categories": [["moveis_decoracao"], ["telefonia"], ["cama_mesa_banho"],
                       ["cama_mesa_banho", "telefonia"], ["beleza_saude"]],
        "purchase_month": ["2017-05", "2017-11", "2018-02", "2018-07", "2018-08"],
        "customer_state": ["SP", "BA", "RJ", "SP", "AM"],
        "customer_region": ["Sudeste", "Nordeste", "Sudeste", "Sudeste", "Norte"],
        "review_creation_date": pd.to_datetime(["2017-06-01", "2017-12-01", "2018-03-01",
                                                "2018-08-01", "2018-09-01"]),
    })


# ------------------------------------------------------------------ filtros


def ids(docs, f):
    return sorted(docs.loc[review_mask(docs, f), "review_id"])


def test_score_and_delivery_filters():
    docs = make_docs()
    assert ids(docs, Filters(score_max=2)) == ["r1", "r2"]
    assert ids(docs, Filters(score_min=4)) == ["r3", "r4"]
    assert ids(docs, Filters(late=True)) == ["r4"]
    assert ids(docs, Filters(late=False)) == ["r1", "r3", "r5"]
    assert ids(docs, Filters(not_received=True)) == ["r2"]


def test_category_period_and_location_filters():
    docs = make_docs()
    assert ids(docs, Filters(categories=["telefonia"])) == ["r2", "r4"]
    assert ids(docs, Filters(period_start="2018-01", period_end="2018-07")) == ["r3", "r4"]
    assert ids(docs, Filters(states=["SP"])) == ["r1", "r4"]
    assert ids(docs, Filters(regions=["Norte"])) == ["r5"]


def test_entry_counts_follow_reviews():
    docs = make_docs()
    counts = entry_counts(docs, review_mask(docs, Filters(score_min=3)), ["d1", "d2", "d3", "d4"])
    assert counts.tolist() == [0, 0, 1, 2]


def test_empty_filters():
    assert Filters().is_empty()
    assert not Filters(score_max=2).is_empty()
    assert Filters().describe() == "sem filtro"


@pytest.mark.parametrize("question, expected", [
    ("Quais problemas são relatados por clientes insatisfeitos?", {"score_max": 2}),
    ("O que os clientes mais elogiam?", {"score_min": 4}),
    ("O que comentam os clientes que deram nota 3?", {"score_min": 3, "score_max": 3}),
    ("O que dizem os clientes quando o pedido não chega?", {"not_received": True}),
    ("Quais temas aparecem nas entregas atrasadas?", {"late": True}),
    ("Por que clientes com entrega no prazo deram nota baixa?", {"late": False, "score_max": 2}),
    ("O que os clientes do Nordeste dizem?", {"regions": ["Nordeste"]}),
    ("Pedidos do Mato Grosso do Sul", {"states": ["MS"]}),
    ("Clientes do Rio Grande do Norte", {"states": ["RN"]}),
    ("Compras de novembro de 2017", {"period_start": "2017-11", "period_end": "2017-11"}),
    ("Reclamações de compras de 2020", {"score_max": 3, "period_start": "2020-01",
                                        "period_end": "2020-12"}),
    ("O que elogiam em informática e acessórios?", {"score_min": 4,
                                                    "categories": ["informatica_acessorios"]}),
])
def test_rule_based_filters(question, expected):
    assert rule_based_filters(question).model_dump(exclude_none=True) == expected


def test_rules_do_not_read_dislike_as_praise():
    assert rule_based_filters("não gostei do produto").score_min is None


# ------------------------------------------------------------------ fusao e diversidade


def test_rrf_rewards_agreement_between_lists():
    fused = rrf([["a", "b", "c"], ["b", "c", "d"]], k=60)
    order = [d for d, _ in fused]
    assert order[0] == "b"
    assert dict(fused)["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert set(order) == {"a", "b", "c", "d"}


def test_mmr_skips_near_duplicates():
    vectors = np.array([[1.0, 0.0], [0.999, 0.045], [0.0, 1.0]])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    order = mmr(np.array([1.0, 0.95, 0.6]), vectors, k=2, lambda_=0.5)
    assert order == [0, 2]


# ------------------------------------------------------------------ recuperacao


class ReverseAlphabetReranker:
    """Re-ranker de teste: ordena pelo texto em ordem alfabetica inversa."""

    def score(self, question, texts):
        ranks = {t: i for i, t in enumerate(sorted(texts))}
        return np.array([ranks[t] for t in texts], dtype=float)


@pytest.fixture()
def retriever() -> Retriever:
    docs = make_docs()
    rep = docs[docs["is_representative"]]
    doc_ids = rep["doc_id"].tolist()
    lexical = LexicalIndex(doc_ids, [tokenize(t) for t in rep["text_clean"]])
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(len(doc_ids), 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return Retriever(docs, lexical, VectorIndex(doc_ids, vectors))


def test_filters_restrict_candidates(retriever):
    qv = retriever.vector.vectors[2]
    hits = retriever.retrieve("produto chegou", Filters(score_min=4), k=4, rerank=False,
                              mmr_lambda=None, query_vector=qv)
    assert {h.doc_id for h in hits} <= {"d3", "d4"}


def test_hit_counts_only_reviews_inside_the_filter(retriever):
    qv = retriever.vector.vectors[3]
    hits = retriever.retrieve("muito bom", Filters(score_min=4), k=4, rerank=False,
                              mmr_lambda=None, query_vector=qv)
    d4 = next(h for h in hits if h.doc_id == "d4")
    assert d4.n_reviews == 1 and d4.review_ids == ["r4"]


def test_hybrid_keeps_ranks_of_both_branches(retriever):
    qv = retriever.vector.vectors[0]
    cands = retriever.candidates("produto quebrado", None, "hybrid", 4, qv)
    top = cands[0]
    assert top["doc_id"] == "d1"
    assert top["rank_lexical"] == 1 and top["rank_vector"] == 1


def test_reranker_reorders_candidates(retriever):
    retriever.reranker = ReverseAlphabetReranker()
    qv = retriever.vector.vectors[0]
    hits = retriever.retrieve("produto chegou", None, k=4, rerank=True, mmr_lambda=None,
                              query_vector=qv)
    texts = [h.text for h in hits]
    assert texts == sorted(texts, reverse=True)
    assert all(h.score_rerank is not None for h in hits)


def test_no_allowed_entries_returns_nothing(retriever):
    hits = retriever.retrieve("produto", Filters(period_start="2020-01"), k=3,
                              query_vector=retriever.vector.vectors[0])
    assert hits == []
