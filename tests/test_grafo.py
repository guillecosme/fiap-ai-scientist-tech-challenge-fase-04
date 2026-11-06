"""Fluxo LangGraph com indice sintetico e LLM falso, e o rastro das consultas."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from src.generation.esquemas import Afirmacao, Analise, Resposta
from src.graph.fluxo import Pipeline
from src.graph.gate import Gate
from src.graph.rastro import load_traces, save_traces, trace_row
from src.indexing.lexical import LexicalIndex
from src.indexing.vetorial import VectorIndex
from src.preprocessing.tokenizacao import tokenize
from src.retrieval.filtros import Filters
from src.retrieval.hibrida import Retriever
from tests.test_recuperacao import make_docs


@dataclass
class Call:
    input_tokens: int = 10
    output_tokens: int = 5
    cached: bool = False


class FakeLLM:
    def __init__(self, analise: Analise | None = None, resposta: Resposta | None = None,
                 fail_on: type | None = None):
        self.analise = analise
        self.resposta = resposta
        self.fail_on = fail_on
        self.calls: list[Call] = []

    def invoke(self, prompt, schema=None, system=None):
        self.calls.append(Call())
        if schema is self.fail_on:
            raise RuntimeError("falha simulada")
        return self.analise if schema is Analise else self.resposta


def analise(escopo="dentro", frequencia=False, filtros=None) -> Analise:
    return Analise(escopo=escopo, frequencia=frequencia, consulta="produto chegou quebrado",
                   filtros=filtros or Filters())


GOOD = Resposta(suficiente=True, resumo="Há relato de produto quebrado.", limitacoes="",
                afirmacoes=[Afirmacao(texto="um cliente recebeu quebrado", evidencias=["E1"]),
                            Afirmacao(texto="afirmação sem base", evidencias=["E42"])])


@pytest.fixture()
def pipeline_factory(monkeypatch):
    docs = make_docs()
    docs["main_category"] = [c[0] for c in docs["categories"]]
    rep = docs[docs["is_representative"]]
    doc_ids = rep["doc_id"].tolist()
    vectors = np.eye(len(doc_ids), 8, dtype=np.float32)
    # o embedding da pergunta aponta para d1 ("produto chegou quebrado")
    monkeypatch.setattr("src.retrieval.hibrida.embed", lambda texts: vectors[[0]])
    retriever = Retriever(docs, LexicalIndex(doc_ids, [tokenize(t) for t in rep["text_clean"]]),
                          VectorIndex(doc_ids, vectors))
    themes = pd.DataFrame({"doc_id": doc_ids, "polaridade": ["problema", "problema", "elogio",
                                                             "elogio"],
                           "topico": [0, 1, 0, 1],
                           "tema": ["Defeito", "Pedido não recebido", "Entrega rápida",
                                    "Elogio curto"]})

    def make(llm, gate=None):
        return Pipeline(retriever, llm, gate=gate or Gate(threshold=-1e9, min_reviews=1),
                        themes=themes, k=3)
    return make


def test_out_of_scope_is_refused_before_retrieval(pipeline_factory):
    p = pipeline_factory(FakeLLM(analise(escopo="fora"), GOOD))
    s = p.run("Qual o faturamento da Olist?")
    assert s["decisao"] == "recusar" and s["motivo"] == "fora_do_escopo"
    assert "hits" not in s
    # a recusa diz o que a base contem, em vez de so negar
    assert "2016 e 2018" in s["resposta"]["resumo"]


def test_answer_keeps_only_cited_claims(pipeline_factory):
    p = pipeline_factory(FakeLLM(analise(), GOOD))
    s = p.run("Há produtos quebrados?")
    assert s["decisao"] == "responder"
    assert [a["texto"] for a in s["resposta"]["afirmacoes"]] == ["um cliente recebeu quebrado"]
    assert s["verificacao"]["validade"] == 0.5
    assert s["evidencias"][0]["id"] == "E1"
    assert s["llm"]["chamadas"] == 2


def test_filter_without_reviews_refuses(pipeline_factory):
    f = Filters(period_start="2020-01", period_end="2020-12")
    p = pipeline_factory(FakeLLM(analise(filtros=f), GOOD))
    s = p.run("Reclamações de 2020?")
    assert s["motivo"] == "sem_avaliacoes" and s["total_recorte"] == 0


def test_gate_threshold_refuses_weak_evidence(pipeline_factory):
    p = pipeline_factory(FakeLLM(analise(), GOOD), gate=Gate(threshold=1e9, min_reviews=1))
    s = p.run("Há produtos quebrados?")
    assert s["motivo"] == "evidencia_insuficiente" and "evidencias" not in s


def test_llm_saying_insufficient_turns_into_refusal(pipeline_factory):
    weak = Resposta(suficiente=False, resumo="", afirmacoes=[], limitacoes="só elogios")
    p = pipeline_factory(FakeLLM(analise(), weak))
    s = p.run("Há produtos quebrados?")
    assert s["decisao"] == "recusar" and s["motivo"] == "llm_insuficiente"
    assert "só elogios" in s["resposta"]["resumo"]


def test_analysis_failure_falls_back_to_rules(pipeline_factory):
    p = pipeline_factory(FakeLLM(None, GOOD, fail_on=Analise))
    s = p.run("Quais problemas os clientes insatisfeitos relatam?")
    assert s["fonte_analise"] == "regras"
    assert s["filtros"] == {"score_max": 2}


def test_generation_failure_returns_evidence(pipeline_factory):
    p = pipeline_factory(FakeLLM(analise(), None, fail_on=Resposta))
    s = p.run("Há produtos quebrados?")
    assert s["decisao"] == "erro_llm" and len(s["evidencias"]) > 0


def test_frequency_is_counted_in_code(pipeline_factory):
    p = pipeline_factory(FakeLLM(analise(frequencia=True, filtros=Filters(score_max=3)), GOOD))
    s = p.run("Quais os principais problemas?")
    freq = {f["tema"]: f["avaliacoes"] for f in s["frequencia"]}
    # notas ate 3: r1 (d1, Defeito), r2 (d2, Pedido nao recebido) e r5 (d4, Elogio curto)
    assert freq == {"Defeito": 1, "Pedido não recebido": 1}


def test_trace_roundtrip(pipeline_factory, tmp_path):
    p = pipeline_factory(FakeLLM(analise(), GOOD))
    s = p.run("Há produtos quebrados?")
    row = trace_row(s, {"question_id": "q99"})
    assert row["decisao"] == "responder" and row["chamadas_llm"] == 2
    path = tmp_path / "rastros.parquet"
    save_traces([row], path)
    save_traces([row], path)
    assert len(load_traces(path)) == 2
