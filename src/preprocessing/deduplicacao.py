"""Deduplicacao de textos identicos com contagem (decisao D04).

Avaliacoes com o mesmo texto normalizado ("muito bom", "otimo", "nao recebi o
produto") formam um grupo e compartilham um `doc_id`. O indice guarda um vetor e uma
entrada lexical por grupo, e a busca devolve o grupo uma vez so, com a contagem de
avaliacoes que ele representa. Assim o top-k nao vira dez copias da mesma frase e a
frequencia real continua disponivel.

Cada avaliacao continua na tabela de documentos com seus proprios metadados: o filtro
por nota, entrega ou categoria e aplicado por avaliacao, e um grupo passa no filtro
quando alguma das suas avaliacoes passa.

Na base da Olist os textos repetidos sao curtos (ver notebook 02); a regra vale para
qualquer tamanho, porque texto identico nao acrescenta evidencia nova.
"""

from __future__ import annotations

import hashlib

import pandas as pd


def doc_id_of(text_norm: str) -> str:
    """Identificador estavel do texto normalizado (12 caracteres do SHA-1)."""
    return hashlib.sha1(text_norm.encode("utf-8")).hexdigest()[:12]


def add_doc_ids(docs: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta doc_id, doc_size (avaliacoes no grupo) e is_representative.

    O representante do grupo e a avaliacao mais antiga, para que a escolha nao dependa
    da ordem das linhas.
    """
    df = docs.copy()
    df["doc_id"] = df["text_norm"].map(doc_id_of)
    df["doc_size"] = df.groupby("doc_id")["review_id"].transform("size").astype(int)
    first = (df.sort_values(["review_creation_date", "review_id"])
             .drop_duplicates("doc_id")["review_id"])
    df["is_representative"] = df["review_id"].isin(set(first))
    return df


def unique_docs(docs: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por doc_id (o representante), na ordem do doc_id: e o que vai ao indice."""
    return (docs[docs["is_representative"]]
            .sort_values("doc_id")
            .reset_index(drop=True))
