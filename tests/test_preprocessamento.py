"""Limpeza, deduplicacao e tokenizacao dos textos.

Emojis aparecem como escapes unicode (\U0001F44D e um polegar para cima).
"""

from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing.deduplicacao import add_doc_ids, unique_docs
from src.preprocessing.limpeza import clean_text, has_content, join_title_comment, normalize
from src.preprocessing.tokenizacao import tokenize


def test_clean_text_handles_line_breaks_emojis_and_repeated_punctuation():
    raw = "Chegou rápido!!!\r\nAmei \U0001F60D\U0001F44D o produto..."
    assert clean_text(raw) == "Chegou rápido! Amei o produto."


def test_clean_text_preserves_accents_and_case():
    assert clean_text("NÃO recebi o pedido") == "NÃO recebi o pedido"


@pytest.mark.parametrize("value", [None, pd.NA, float("nan")])
def test_clean_text_missing_values(value):
    assert clean_text(value) == ""


def test_normalize_removes_accents_case_and_emphasis():
    assert normalize("Ótimo!!! Muitoooo bom \U0001F44D") == "otimo muito bom"
    assert normalize("Não   RECEBI") == normalize("nao recebi")


def test_has_content_requires_a_letter():
    assert not has_content(normalize("......"))
    assert not has_content(normalize("10"))
    assert not has_content(normalize("\U0001F44D"))
    assert has_content(normalize("ok"))


def test_join_title_comment():
    joined = join_title_comment("Produto errado", "mandaram outro")
    assert joined == "Produto errado. mandaram outro"
    assert join_title_comment("Ótimo!", "chegou cedo") == "Ótimo! chegou cedo"
    assert join_title_comment(None, "chegou cedo") == "chegou cedo"
    assert join_title_comment("Recomendo", None) == "Recomendo"


def test_dedup_groups_identical_normalized_texts():
    docs = pd.DataFrame({
        "review_id": ["a", "b", "c", "d"],
        "text_norm": ["muito bom", "muito bom", "nao recebi", "muito bom"],
        "review_creation_date": pd.to_datetime(["2018-03-01", "2018-01-01", "2018-02-01",
                                                "2018-05-01"]),
    })
    out = add_doc_ids(docs).set_index("review_id")
    assert out.loc["a", "doc_id"] == out.loc["b", "doc_id"] == out.loc["d", "doc_id"]
    assert out.loc["a", "doc_id"] != out.loc["c", "doc_id"]
    assert out.loc["a", "doc_size"] == 3 and out.loc["c", "doc_size"] == 1
    # o representante e a avaliacao mais antiga do grupo
    assert out.loc["b", "is_representative"] and not out.loc["a", "is_representative"]
    assert len(unique_docs(out.reset_index())) == 2


def test_tokenize_keeps_negations_and_drops_stopwords():
    tokens = tokenize("Eu não recebi o produto")
    assert "nao" in tokens
    assert "eu" not in tokens and "o" not in tokens
    assert tokenize("não recebi") != tokenize("recebi")


def test_tokenize_is_accent_insensitive_and_stems():
    assert tokenize("ótimo") == tokenize("otimo")
    assert tokenize("entrega")[0] == tokenize("entregue")[0] == tokenize("entregaram")[0]
    assert tokenize("muitooo atrasado") == tokenize("atrasado")


DOCS = Path("data/processed/documentos.parquet")


@pytest.mark.skipif(not DOCS.exists(), reason="documentos ausentes")
def test_documents_contract():
    df = pd.read_parquet(DOCS)
    assert df["review_id"].is_unique
    assert df["text_norm"].map(has_content).all()
    assert (df.groupby("doc_id")["text_norm"].nunique() == 1).all()
    assert df["is_representative"].sum() == df["doc_id"].nunique()
    assert df["tokens"].str.len().gt(0).all()
    assert df["n_chars"].max() <= 300
