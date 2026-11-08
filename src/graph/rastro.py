"""Rastro das consultas em Parquet (D15).

Cada consulta vira uma linha com tudo que explica a resposta: a analise da pergunta, os
filtros, as entradas recuperadas com scores, a decisao do gate, a resposta, a validade
das citacoes, o tempo de cada etapa e os tokens. O rastro permite auditar uma resposta
depois e e a base das metricas de operacao (latencia e custo) do notebook 06.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

TRACES = Path("reports/rastros/rastros.parquet")


def trace_row(state: dict, extra: dict | None = None) -> dict:
    analise = state.get("analise") or {}
    resposta = state.get("resposta") or {}
    verificacao = state.get("verificacao") or {}
    llm = state.get("llm") or {}
    hits = [{"doc_id": h["doc_id"], "score": round(h["score"], 4),
             "n_reviews": h["n_reviews"], "review_ids": h["review_ids"][:5]}
            for h in state.get("hits", [])]
    row = {
        "id": state.get("id"),
        "pergunta": state.get("pergunta"),
        "fonte_analise": state.get("fonte_analise"),
        "escopo": analise.get("escopo"),
        "pergunta_de_frequencia": analise.get("frequencia"),
        "consulta": analise.get("consulta"),
        "filtros": json.dumps(state.get("filtros", {}), ensure_ascii=False),
        "recorte": state.get("recorte"),
        "total_recorte": state.get("total_recorte"),
        "n_hits": len(hits),
        "hits": json.dumps(hits, ensure_ascii=False),
        "frequencia": json.dumps([{k: f[k] for k in ("tema", "avaliacoes", "pct")}
                                  for f in state.get("frequencia") or []], ensure_ascii=False),
        "top_score": (state.get("gate") or {}).get("top_score"),
        "decisao": state.get("decisao"),
        "motivo": state.get("motivo"),
        "resumo": resposta.get("resumo"),
        "resposta": json.dumps(resposta, ensure_ascii=False),
        "n_afirmacoes": len(resposta.get("afirmacoes", [])),
        "validade_citacoes": verificacao.get("validade"),
        "afirmacoes_removidas": len(verificacao.get("removidas", [])),
        "tempos": json.dumps({k: round(v, 3) for k, v in state.get("tempos", {}).items()}),
        "latencia_s": state.get("tempos", {}).get("total"),
        "chamadas_llm": llm.get("chamadas"),
        "latencia_llm_s": llm.get("latencia_llm_s"),
        "tokens_entrada": llm.get("tokens_entrada"),
        "tokens_saida": llm.get("tokens_saida"),
        "erros": json.dumps(state.get("erros", []), ensure_ascii=False),
    }
    return {**row, **(extra or {})}


def save_traces(rows: list[dict], path: Path = TRACES, append: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if append and path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def load_traces(path: Path = TRACES) -> pd.DataFrame:
    return pd.read_parquet(path)
