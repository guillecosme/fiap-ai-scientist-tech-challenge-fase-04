"""Documentos do indice: uma linha por avaliacao com texto.

Uso:
    python -m src.preprocessing.documentos   # le a base e grava data/processed/documentos.parquet

Cada avaliacao com titulo ou comentario vira um documento (sem fragmentacao, D01): o
comentario mais longo da base tem 208 caracteres. O texto original fica preservado para
exibicao; o texto limpo e o normalizado sao gerados por `src.preprocessing.limpeza`.
Avaliacoes cujo texto nao tem nenhuma letra depois de normalizado ficam fora. Textos
identicos depois de normalizados compartilham um doc_id (`src.preprocessing.deduplicacao`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.preprocessing.deduplicacao import add_doc_ids
from src.preprocessing.limpeza import clean_text, has_content, join_title_comment, normalize

BASE = Path("data/processed/avaliacoes.parquet")
OUTPUT = Path("data/processed/documentos.parquet")

METADATA = [
    "review_score", "review_creation_date", "review_answer_timestamp", "order_ids",
    "purchase_year", "purchase_month", "delivery_status", "delay_days", "is_late",
    "answered_before_delivery", "main_category", "categories", "customer_state",
    "customer_region",
]


def build_documents(base: pd.DataFrame) -> pd.DataFrame:
    df = base[base["has_comment"] | base["has_title"]].copy()
    df["title"] = df["review_comment_title"].map(clean_text)
    df["comment"] = df["review_comment_message"].map(clean_text)
    df["text_clean"] = [join_title_comment(t, c) for t, c in
                        zip(df["review_comment_title"], df["review_comment_message"], strict=True)]
    df["text_norm"] = df["text_clean"].map(normalize)
    df = df[df["text_norm"].map(has_content)]
    df["n_chars"] = df["text_clean"].str.len()
    df["n_words"] = df["text_clean"].str.split().str.len()
    cols = ["review_id", "review_comment_title", "review_comment_message", "text_clean",
            "text_norm", "n_chars", "n_words"] + METADATA
    return add_doc_ids(df[cols].reset_index(drop=True))


def main() -> None:
    docs = build_documents(pd.read_parquet(BASE))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    docs.to_parquet(OUTPUT, index=False, compression="zstd")
    print(f"{len(docs):,} avaliacoes com texto, {docs['doc_id'].nunique():,} textos distintos "
          f"-> {OUTPUT}")


if __name__ == "__main__":
    main()
