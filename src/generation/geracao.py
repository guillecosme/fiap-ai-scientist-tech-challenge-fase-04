"""Geracao da resposta fundamentada: contexto, cadeia LCEL e saida estruturada.

A cadeia e `PROMPT | LLM com schema Resposta`. O LLM recebe as evidencias numeradas e,
nas perguntas de frequencia, a contagem por tema feita em codigo (D10). Cada afirmacao
volta com os ids das evidencias; `src.generation.citacoes` confere esses ids depois.
"""

from __future__ import annotations

import pandas as pd
from langchain_core.runnables import RunnableLambda

from src.generation.esquemas import Resposta
from src.generation.prompts import PROMPT, format_evidence, format_frequency
from src.visualization.rotulos import DELIVERY_LABELS, category_label


def build_evidence(hits, docs: pd.DataFrame) -> list[dict]:
    """Evidencias E1..En a partir dos resultados da busca, com metadados para exibicao."""
    by_review = docs.set_index("review_id")
    evidence = []
    for i, h in enumerate(hits, start=1):
        rid = h.review_ids[0] if h.review_ids else None
        row = by_review.loc[rid] if rid is not None else None
        evidence.append({
            "id": f"E{i}",
            "doc_id": h.doc_id,
            "review_ids": list(h.review_ids),
            "texto": h.text,
            "nota": int(row["review_score"]) if row is not None else None,
            "data": str(row["review_creation_date"].date()) if row is not None else "",
            "categoria": category_label(row["main_category"]) if row is not None else "",
            "entrega": DELIVERY_LABELS.get(row["delivery_status"], "") if row is not None
            else "",
            "n_avaliacoes": int(h.n_reviews),
            "score": float(h.score),
        })
    return evidence


def generation_chain(llm):
    """Runnable LCEL: dicionario de variaveis -> Resposta."""
    return PROMPT | RunnableLambda(lambda value: llm.invoke(value.to_messages(),
                                                            schema=Resposta))


def generate(llm, question: str, evidence: list[dict], recorte: str,
             freq: pd.DataFrame | None = None, total_reviews: int = 0) -> Resposta:
    return generation_chain(llm).invoke({
        "pergunta": question,
        "recorte": recorte,
        "frequencia": format_frequency(freq, total_reviews, recorte),
        "evidencias": format_evidence(evidence),
    })
