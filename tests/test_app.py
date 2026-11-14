"""Interface Streamlit: funcoes de apresentacao, auditoria e a tela com pipeline falso."""

from types import SimpleNamespace

import pandas as pd
import pytest

from src.app import servicos

STATE = {
    "id": "x", "pergunta": "Quais os problemas?", "decisao": "responder", "motivo": None,
    "recorte": "nota de 1 a 3", "total_recorte": 14431,
    "analise": {"consulta": "atraso produto errado"}, "fonte_analise": "llm",
    "filtros": {"score_max": 3},
    "resposta": {"suficiente": True, "resumo": "Atrasos e produtos errados.", "limitacoes": "",
                 "afirmacoes": [{"texto": "Atraso é o mais citado", "evidencias": ["E2"]}]},
    "evidencias": [
        {"id": "E1", "texto": "não citada", "nota": 1, "data": "2018-01-01", "categoria": "A",
         "entrega": "Entregue no prazo", "n_avaliacoes": 1, "review_ids": ["r1"]},
        {"id": "E2", "texto": "chegou atrasado", "nota": 2, "data": "2018-01-02",
         "categoria": "Telefonia", "entrega": "Entregue com atraso", "n_avaliacoes": 3,
         "review_ids": ["r2", "r3", "r4"]},
    ],
    "frequencia": [{"polaridade": "problema", "tema": "Pedido não recebido",
                    "avaliacoes": 5551, "pct": 39.7}],
    "hits": [{"doc_id": "d2", "text": "chegou atrasado", "score": 3.2, "n_reviews": 3,
              "review_ids": ["r2"], "rank_lexical": 1, "rank_vector": 4, "score_rerank": 3.2}],
    "tempos": {"analisar_llm": 1.5, "recuperar": 1.0, "gerar_llm": 3.0, "total": 5.6},
    "llm": {"chamadas": 2},
}


def test_decision_titles():
    assert servicos.decision(STATE)[1] == "success"
    recusa = servicos.decision({"decisao": "recusar", "motivo": "fora_do_escopo"})
    assert recusa[1] == "warning" and "fora do escopo" in recusa[0]
    assert servicos.decision({"decisao": "erro_llm"})[1] == "error"


def test_only_cited_evidence_is_shown():
    assert [e["id"] for e in servicos.cited_evidence(STATE)] == ["E2"]
    falha = {**STATE, "decisao": "erro_llm"}
    assert len(servicos.cited_evidence(falha)) == 2


def test_frequency_retrieval_and_times():
    freq = servicos.frequency_table(STATE)
    assert freq.loc[0, "Tema"] == "Pedido não recebido" and freq.loc[0, "Avaliações"] == 5551
    assert servicos.frequency_table({}).empty
    assert servicos.retrieval_table(STATE).loc[0, "posição BM25"] == 1
    assert servicos.stage_times(STATE) == {"análise (LLM)": 1.5, "recuperação": 1.0,
                                           "geração (LLM)": 3.0}


def test_formatting():
    assert servicos.score_label(2) == "nota 2 de 5"
    assert servicos.br(-0.3688) == "-0,37"


@pytest.fixture()
def audit_file(tmp_path):
    path = tmp_path / "auditoria.csv"
    pd.DataFrame({"question_id": ["q1", "q1", "q2"], "pergunta": ["p", "p", "p2"],
                  "doc_id": ["a", "b", "c"], "texto": ["t1", "t2", "t3"],
                  "nota_juiz": [2, 0, 1], "nota_segundo_juiz": [2, 1, 1],
                  "nota_manual": ["", "", ""]}).to_csv(path, index=False)
    return path


def test_audit_labels_and_agreement(audit_file):
    df = servicos.load_audit(audit_file)
    assert servicos.next_unlabeled(df) == 0
    assert servicos.audit_agreement(df) == {"rotulados": 0}
    df = servicos.save_label(df, 0, 2, audit_file)
    df = servicos.save_label(df, 1, 1, audit_file)
    reloaded = servicos.load_audit(audit_file)
    assert servicos.next_unlabeled(reloaded) == 2
    acordo = servicos.audit_agreement(reloaded)
    assert acordo["rotulados"] == 2
    assert acordo["concordancia_juiz"] == 0.5 and acordo["concordancia_segundo_juiz"] == 1.0
    with pytest.raises(ValueError):
        servicos.save_label(reloaded, 2, 5, audit_file)


class FakePipeline:
    def __init__(self):
        self.retriever = SimpleNamespace(retrieve=lambda *a, **k: [])
        self.gate = SimpleNamespace(threshold=-0.37)
        self.llm = SimpleNamespace(model="modelo-teste")

    def run(self, pergunta):
        return {**STATE, "pergunta": pergunta}


def test_app_runs_a_question(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(servicos, "build_pipeline", lambda: (FakePipeline(), None))
    at = AppTest.from_file("../app.py", default_timeout=60)
    at.run()
    assert not at.exception
    at.selectbox[0].set_value("Problema de negócio").run()
    next(b for b in at.button if b.label == "Perguntar").click().run()
    assert not at.exception
    assert "Resposta fundamentada" in at.success[0].value
    assert any("chegou atrasado" in m.value for m in at.markdown)
