"""API FastAPI: pergunta em linguagem natural, resposta com evidencias.

Uso:
    make api        # sobe em http://127.0.0.1:8000; documentacao interativa em /docs

Endpoints:
- GET /saude: estado da API e dos indices carregados;
- POST /perguntar: executa o fluxo LangGraph e devolve a decisao (responder ou recusar),
  o resumo, as afirmacoes com os ids das evidencias, as evidencias com os metadados e o
  rastro da consulta (filtros, tempos, chamadas ao LLM).

O pipeline (indices, modelos e LLM) e carregado uma vez, na subida da API, com uma busca
de aquecimento para que a primeira pergunta nao pague a carga dos modelos.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.graph.rastro import save_traces, trace_row

API_TRACES = "reports/rastros/api.parquet"


class Pergunta(BaseModel):
    pergunta: str = Field(min_length=3, max_length=500,
                          examples=["Quais são os principais problemas relatados pelos "
                                    "clientes?"])


class Evidencia(BaseModel):
    id: str
    texto: str
    nota: int | None
    data: str
    categoria: str
    entrega: str
    avaliacoes_com_o_mesmo_texto: int
    review_ids: list[str]


class AfirmacaoOut(BaseModel):
    texto: str
    evidencias: list[str]


class Tema(BaseModel):
    tema: str
    avaliacoes: int
    pct: float


class RespostaOut(BaseModel):
    id: str
    pergunta: str
    decisao: str = Field(description="responder, recusar ou erro_llm")
    motivo: str | None = Field(None, description="motivo da recusa, quando houver")
    resumo: str
    afirmacoes: list[AfirmacaoOut]
    limitacoes: str
    recorte: str | None
    avaliacoes_no_recorte: int | None
    frequencia_por_tema: list[Tema]
    evidencias: list[Evidencia]
    tempos_s: dict[str, float]
    chamadas_llm: int | None


def to_response(state: dict) -> RespostaOut:
    resposta = state.get("resposta") or {}
    cited = {c for a in resposta.get("afirmacoes", []) for c in a["evidencias"]}
    evidencias = [e for e in state.get("evidencias", [])
                  if state.get("decisao") == "erro_llm" or e["id"] in cited]
    return RespostaOut(
        id=state.get("id", ""),
        pergunta=state["pergunta"],
        decisao=state.get("decisao", "recusar"),
        motivo=state.get("motivo"),
        resumo=resposta.get("resumo", ""),
        afirmacoes=[AfirmacaoOut(**a) for a in resposta.get("afirmacoes", [])],
        limitacoes=resposta.get("limitacoes", ""),
        recorte=state.get("recorte"),
        avaliacoes_no_recorte=state.get("total_recorte"),
        frequencia_por_tema=[Tema(tema=f["tema"], avaliacoes=f["avaliacoes"], pct=f["pct"])
                             for f in (state.get("frequencia") or [])[:8]],
        evidencias=[Evidencia(id=e["id"], texto=e["texto"], nota=e["nota"], data=e["data"],
                              categoria=e["categoria"], entrega=e["entrega"],
                              avaliacoes_com_o_mesmo_texto=e["n_avaliacoes"],
                              review_ids=e["review_ids"]) for e in evidencias],
        tempos_s={k: round(v, 3) for k, v in state.get("tempos", {}).items()},
        chamadas_llm=(state.get("llm") or {}).get("chamadas"),
    )


_pipeline = None


def build_pipeline():
    from src.generation.provedor import LLM
    from src.graph.fluxo import Pipeline
    from src.graph.gate import Gate
    from src.retrieval.hibrida import Retriever
    from src.retrieval.reranking import CrossEncoderReranker

    return Pipeline(Retriever.load(CrossEncoderReranker()), LLM("chat"), gate=Gate.load())


def get_pipeline():
    if _pipeline is None:
        raise HTTPException(503, "pipeline ainda carregando")
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    if os.environ.get("TC_API_SEM_PIPELINE") != "1":
        _pipeline = build_pipeline()
        # carrega os modelos de embedding e de re-ranking antes da primeira pergunta
        _pipeline.retriever.retrieve("entrega", k=1)
    yield


app = FastAPI(title="Voice of Customer Intelligence com RAG",
              description="Perguntas sobre as avaliações de clientes da Olist, respondidas "
                          "com evidências rastreáveis.",
              version="1.0.0", lifespan=lifespan)


@app.get("/saude")
def saude():
    return {"status": "ok", "pipeline_carregado": _pipeline is not None}


@app.post("/perguntar", response_model=RespostaOut)
def perguntar(body: Pergunta,
              pipeline: Annotated[Any, Depends(get_pipeline)]) -> RespostaOut:
    state = pipeline.run(body.pergunta.strip())
    if os.environ.get("TC_API_RASTRO", "1") == "1":
        from pathlib import Path

        save_traces([trace_row(state)], Path(API_TRACES))
    return to_response(state)
