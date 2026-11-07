"""Fluxo da consulta em LangGraph (D14).

    analisar -> (fora do escopo) -> recusar
             -> recuperar -> avaliar_evidencia -> (insuficiente) -> recusar
                                               -> gerar -> verificar -> fim

Um grafo com arestas condicionais explicitas, em vez de um agente que decide sozinho
quais ferramentas chamar: o caminho de cada pergunta fica auditavel no rastro, o numero
de chamadas ao LLM e fixo (no maximo duas) e a recusa e uma decisao de codigo.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from src.generation.citacoes import verify
from src.generation.esquemas import Analise, Resposta
from src.generation.geracao import build_evidence, generate
from src.graph.gate import Gate
from src.indexing.temas import load_themes, theme_frequency
from src.retrieval.filtros import Filters, review_mask, rule_based_filters

ANALYSIS_SYSTEM = (
    "Você prepara perguntas de gestores de e-commerce para uma busca em avaliações de "
    "clientes da Olist (pedidos de 2016 a 2018, Brasil). Preencha o schema. Use filtros só "
    "quando a pergunta pedir um recorte explícito: nota (insatisfeitos ou nota baixa = "
    "nota máxima 2; reclamações ou problemas = nota máxima 3; elogios ou satisfeitos = "
    "nota mínima 4), atraso, pedido não recebido, entrega no prazo, categoria de produto "
    "(nome interno da Olist, como cama_mesa_banho, informatica_acessorios, telefonia, "
    "beleza_saude, moveis_decoracao, esporte_lazer, relogios_presentes, "
    "utilidades_domesticas, brinquedos, automotivo), período da compra (AAAA-MM), UF ou "
    "região. Um período fora de 2016 a 2018 deve ser preenchido mesmo assim."
)

PROBLEM_WORDS = r"\b(problema|reclama|insatisf|ruim|negativ|nota baixa|falha|defeito)"
FREQUENCY_WORDS = r"\b(principa|frequen|mais comu|recorrent|padr|ranking|mais |quais )"

BASE_SCOPE = ("A base reúne notas e comentários de clientes sobre pedidos da Olist feitos entre "
              "2016 e 2018: entrega, produto, atendimento, loja e elogios.")

REFUSALS = {
    "fora_do_escopo": "Não é possível responder com as avaliações de clientes: a pergunta pede "
                      "uma informação que elas não contêm (métricas internas, financeiras ou "
                      "assuntos alheios à experiência de compra). " + BASE_SCOPE,
    "sem_avaliacoes": "Nenhuma avaliação com comentário atende ao recorte pedido ({recorte}). "
                      + BASE_SCOPE + " Tente ampliar o recorte.",
    "evidencia_insuficiente": "As avaliações encontradas não tratam do assunto perguntado com "
                              "relevância suficiente para sustentar uma resposta. Não há "
                              "evidência nos comentários para afirmar algo sobre isso.",
    "poucas_avaliacoes": "Há só {n} avaliação(ões) com comentário no recorte pedido "
                         "({recorte}), poucas para identificar um padrão.",
    "llm_insuficiente": "As evidências recuperadas não bastam para responder: {detalhe}",
}


class Estado(TypedDict, total=False):
    id: str
    pergunta: str
    analise: dict
    fonte_analise: str
    filtros: dict
    recorte: str
    total_recorte: int
    hits: list[dict]
    frequencia: list[dict]
    gate: dict
    decisao: str
    motivo: str
    evidencias: list[dict]
    resposta: dict
    verificacao: dict
    tempos: dict
    erros: list[str]


class Pipeline:
    """Recursos do fluxo (indices, modelos, LLM) e os nos do grafo."""

    def __init__(self, retriever, llm=None, gate: Gate | None = None,
                 themes: pd.DataFrame | None = None, k: int = 10):
        self.retriever = retriever
        self.docs = retriever.docs
        self.llm = llm
        self.gate = gate or Gate()
        self.themes = themes if themes is not None else load_themes()
        self.k = k
        self.graph = self._build()

    @classmethod
    def load(cls, k: int = 10) -> Pipeline:
        """Pipeline completo: indices, cross-encoder, LLM de geracao e gate calibrado."""
        from src.generation.provedor import LLM
        from src.retrieval.hibrida import Retriever
        from src.retrieval.reranking import CrossEncoderReranker

        return cls(Retriever.load(CrossEncoderReranker()), LLM("chat"), gate=Gate.load(), k=k)

    # ------------------------------------------------------------------ nos

    def analisar(self, s: Estado) -> Estado:
        import re

        t0 = time.perf_counter()
        q = s["pergunta"]
        erros = list(s.get("erros", []))
        try:
            if self.llm is None:
                raise RuntimeError("sem LLM configurado")
            a = self.llm.invoke(q, schema=Analise, system=ANALYSIS_SYSTEM)
            analise, fonte = a.model_dump(), "llm"
            llm_time = {"analisar_llm": self.llm.calls[-1].latency_s}
        except Exception as exc:  # noqa: BLE001
            erros.append(f"analise: {type(exc).__name__}")
            low = q.lower()
            analise = {"escopo": "dentro",
                       "frequencia": bool(re.search(FREQUENCY_WORDS, low)),
                       "consulta": q, "filtros": rule_based_filters(q).model_dump()}
            fonte = "regras"
            llm_time = {}
        filtros = Filters(**analise["filtros"])
        return {"analise": analise, "fonte_analise": fonte,
                "filtros": filtros.model_dump(exclude_none=True),
                "recorte": filtros.describe(), "erros": erros,
                "tempos": {**s.get("tempos", {}), "analisar": time.perf_counter() - t0,
                           **llm_time}}

    def recuperar(self, s: Estado) -> Estado:
        t0 = time.perf_counter()
        filtros = Filters(**s["filtros"])
        consulta = s["analise"].get("consulta") or s["pergunta"]
        hits = self.retriever.retrieve(consulta, filtros, k=self.k)
        mask = (review_mask(self.docs, filtros) if not filtros.is_empty()
                else pd.Series(True, index=self.docs.index))
        freq = []
        if s["analise"].get("frequencia") and mask.any():
            f = theme_frequency(self.docs, self.themes, mask)
            side = self._side(filtros)
            if side:
                f = f[f["polaridade"] == side]
                f["pct"] = (100 * f["avaliacoes"] / f["avaliacoes"].sum()).round(1)
            freq = f.to_dict("records")
        return {"hits": [asdict(h) for h in hits], "total_recorte": int(mask.sum()),
                "frequencia": freq,
                "tempos": {**s["tempos"], "recuperar": time.perf_counter() - t0}}

    def avaliar_evidencia(self, s: Estado) -> Estado:
        result = self.gate.decide(s["hits"], s["total_recorte"])
        out: Estado = {"gate": result}
        out["decisao"] = "responder" if result["passa"] else "recusar"
        if not result["passa"]:
            out["motivo"] = result["motivo"]
        return out

    def gerar(self, s: Estado) -> Estado:
        from src.retrieval.hibrida import Hit

        t0 = time.perf_counter()
        hits = [Hit(**h) for h in s["hits"]]
        evidence = build_evidence(hits, self.docs)
        freq = pd.DataFrame(s.get("frequencia") or [])
        try:
            resposta = generate(self.llm, s["pergunta"], evidence, s["recorte"], freq,
                                s["total_recorte"]).model_dump()
        except Exception as exc:  # noqa: BLE001
            # sem resposta do LLM, o gestor ainda recebe as evidencias encontradas
            resposta = {"suficiente": True, "afirmacoes": [], "limitacoes": "",
                        "resumo": "A resposta não pôde ser gerada agora (falha no serviço "
                                  "do modelo de linguagem). As avaliações recuperadas para a "
                                  "pergunta estão listadas como evidência."}
            return {"evidencias": evidence, "resposta": resposta, "decisao": "erro_llm",
                    "erros": [*s.get("erros", []), f"gerar: {type(exc).__name__}"],
                    "tempos": {**s["tempos"], "gerar": time.perf_counter() - t0}}
        return {"evidencias": evidence, "resposta": resposta,
                "tempos": {**s["tempos"], "gerar": time.perf_counter() - t0,
                           "gerar_llm": self.llm.calls[-1].latency_s}}

    def verificar(self, s: Estado) -> Estado:
        resposta = Resposta(**s["resposta"])
        checked, v = verify(resposta, {e["id"] for e in s["evidencias"]})
        out: Estado = {"resposta": checked.model_dump(),
                       "verificacao": {**asdict(v), "validade": v.validade,
                                       "validade_citacoes": v.validade_citacoes}}
        if not checked.suficiente:
            out["decisao"] = "recusar"
            out["motivo"] = "llm_insuficiente"
        return out

    def recusar(self, s: Estado) -> Estado:
        motivo = s.get("motivo") or "fora_do_escopo"
        detalhe = (s.get("resposta") or {}).get("limitacoes") or "faltam avaliações sobre o tema"
        texto = REFUSALS[motivo].format(recorte=s.get("recorte", "sem filtro"),
                                        n=s.get("total_recorte", 0), detalhe=detalhe)
        return {"decisao": "recusar", "motivo": motivo,
                "resposta": {"suficiente": False, "resumo": texto, "afirmacoes": [],
                             "limitacoes": texto}}

    # ------------------------------------------------------------------ grafo

    @staticmethod
    def _side(f: Filters) -> str | None:
        if f.score_max is not None and f.score_max <= 3:
            return "problema"
        if f.score_min is not None and f.score_min >= 4:
            return "elogio"
        return None

    def _after_gate(self, s: Estado) -> str:
        if s["decisao"] == "responder" and self.llm is not None:
            return "gerar"
        return "recusar"

    def _build(self):
        g = StateGraph(Estado)
        for name in ["analisar", "recuperar", "avaliar_evidencia", "gerar", "verificar",
                     "recusar"]:
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "analisar")
        g.add_conditional_edges("analisar", lambda s: "recusar" if s["analise"][
            "escopo"] == "fora" else "recuperar", ["recuperar", "recusar"])
        g.add_edge("recuperar", "avaliar_evidencia")
        g.add_conditional_edges("avaliar_evidencia", self._after_gate, ["gerar", "recusar"])
        g.add_conditional_edges("gerar", lambda s: END if s.get("decisao") == "erro_llm"
                                else "verificar", ["verificar", END])
        g.add_conditional_edges("verificar", lambda s: "recusar" if s.get(
            "decisao") == "recusar" else END, ["recusar", END])
        g.add_edge("recusar", END)
        return g.compile()

    def run(self, pergunta: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        calls_before = len(self.llm.calls) if self.llm is not None else 0
        state = self.graph.invoke({"id": uuid.uuid4().hex[:12], "pergunta": pergunta,
                                   "tempos": {}, "erros": []})
        state["tempos"]["total"] = time.perf_counter() - t0
        if self.llm is not None:
            new = self.llm.calls[calls_before:]
            state["llm"] = {"chamadas": len(new),
                            "latencia_llm_s": sum(c.latency_s for c in new),
                            "tokens_entrada": sum(c.input_tokens for c in new),
                            "tokens_saida": sum(c.output_tokens for c in new),
                            "em_cache": sum(c.cached for c in new)}
        return state
