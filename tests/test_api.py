"""Contrato da API com um pipeline falso (sem indices nem LLM)."""

import os

import pytest

os.environ["TC_API_SEM_PIPELINE"] = "1"
os.environ["TC_API_RASTRO"] = "0"

from fastapi.testclient import TestClient  # noqa: E402

from src.api import main  # noqa: E402

ANSWER = {
    "id": "abc", "pergunta": "Quais os problemas?", "decisao": "responder", "motivo": None,
    "recorte": "nota de 1 a 3", "total_recorte": 120,
    "resposta": {"suficiente": True, "resumo": "Atrasos e produtos errados.",
                 "limitacoes": "", "afirmacoes": [{"texto": "Atraso", "evidencias": ["E1"]}]},
    "evidencias": [
        {"id": "E1", "texto": "chegou atrasado", "nota": 1, "data": "2018-01-02",
         "categoria": "Telefonia", "entrega": "Entregue com atraso", "n_avaliacoes": 2,
         "review_ids": ["r1", "r2"], "doc_id": "d1", "score": 3.1},
        {"id": "E2", "texto": "não citada", "nota": 2, "data": "2018-01-03",
         "categoria": "Brinquedos", "entrega": "Entregue no prazo", "n_avaliacoes": 1,
         "review_ids": ["r3"], "doc_id": "d2", "score": 1.0},
    ],
    "frequencia": [{"polaridade": "problema", "tema": "Pedido não recebido",
                    "avaliacoes": 50, "pct": 41.7}],
    "tempos": {"analisar": 1.0, "total": 3.5}, "llm": {"chamadas": 2},
}
REFUSAL = {"id": "def", "pergunta": "Qual o faturamento?", "decisao": "recusar",
           "motivo": "fora_do_escopo", "tempos": {"total": 0.8},
           "resposta": {"suficiente": False, "resumo": "Fora do escopo.", "afirmacoes": [],
                        "limitacoes": "Fora do escopo."}}


class FakePipeline:
    def __init__(self, state):
        self.state = state

    def run(self, pergunta):
        return {**self.state, "pergunta": pergunta}


@pytest.fixture()
def client():
    with TestClient(main.app) as c:
        yield c
    main.app.dependency_overrides.clear()


def use(state):
    main.app.dependency_overrides[main.get_pipeline] = lambda: FakePipeline(state)


def test_health(client):
    r = client.get("/saude")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_answer_contract(client):
    use(ANSWER)
    r = client.post("/perguntar", json={"pergunta": "Quais os problemas?"})
    assert r.status_code == 200
    body = r.json()
    assert body["decisao"] == "responder"
    assert body["afirmacoes"][0]["evidencias"] == ["E1"]
    # so as evidencias citadas voltam para o gestor
    assert [e["id"] for e in body["evidencias"]] == ["E1"]
    assert body["evidencias"][0]["avaliacoes_com_o_mesmo_texto"] == 2
    assert body["frequencia_por_tema"][0]["tema"] == "Pedido não recebido"
    assert body["avaliacoes_no_recorte"] == 120


def test_refusal_contract(client):
    use(REFUSAL)
    body = client.post("/perguntar", json={"pergunta": "Qual o faturamento?"}).json()
    assert body["decisao"] == "recusar" and body["motivo"] == "fora_do_escopo"
    assert body["afirmacoes"] == [] and body["evidencias"] == []


def test_rejects_empty_question(client):
    use(ANSWER)
    assert client.post("/perguntar", json={"pergunta": ""}).status_code == 422
    assert client.post("/perguntar", json={}).status_code == 422


def test_unavailable_without_pipeline(client):
    assert client.post("/perguntar", json={"pergunta": "teste"}).status_code == 503
