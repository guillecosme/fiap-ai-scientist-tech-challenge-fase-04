"""Funcoes da interface Streamlit, separadas da tela para poderem ser testadas.

A interface usa o mesmo `Pipeline` da API (`src/graph/fluxo.py`); aqui ficam a carga do
pipeline, a traducao do estado final do grafo para o que a tela mostra, o resumo das
metricas de qualidade e a auditoria manual do juiz.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.julgamento import AUDIT, cohen_kappa

TRACES = Path("reports/rastros/avaliacao.parquet")
ABLATION = Path("reports/ablacao_recuperacao.csv")
FAITHFULNESS = Path("data/eval/fidelidade.parquet")
THRESHOLD = Path("data/eval/limiar_evidencia.json")
METRICS = Path("reports/metricas.md")

EXEMPLOS = {
    "Problema de negócio": [
        "Quais são os principais problemas relatados pelos clientes?",
        "Quais reclamações relacionadas à entrega aparecem com maior frequência?",
        "O que os clientes mais elogiam em suas experiências de compra?",
        "Quais problemas são relatados por clientes insatisfeitos?",
        "Existem padrões recorrentes nas avaliações relacionadas à qualidade dos produtos?",
    ],
    "Exploração": [
        "O que os clientes relatam sobre problemas relacionados à entrega?",
        "Quais avaliações sustentam a conclusão de que existem problemas recorrentes "
        "relacionados ao prazo de entrega?",
    ],
    "Metadados": [
        "Quais são os principais problemas relatados nas avaliações com nota baixa?",
        "O que os clientes mais elogiam na categoria cama, mesa e banho?",
        "Quais temas aparecem nas avaliações de entregas atrasadas?",
    ],
    "Devem ser recusadas": [
        "Qual foi o faturamento total da Olist em 2018?",
        "O que os clientes acham do pagamento via PIX?",
        "O que os clientes falam dos produtos da categoria seguros e serviços?",
    ],
}

MOTIVOS = {
    "fora_do_escopo": "fora do escopo da base",
    "sem_avaliacoes": "nenhuma avaliação no recorte pedido",
    "poucas_avaliacoes": "poucas avaliações no recorte pedido",
    "evidencia_insuficiente": "evidência sem relevância suficiente",
    "llm_insuficiente": "o modelo julgou as evidências insuficientes",
}


def build_pipeline():
    """Pipeline completo; sem LLM configurado, analisa por regras e recusa a geracao."""
    from src.graph.fluxo import Pipeline

    try:
        return Pipeline.load(), None
    except RuntimeError as exc:
        from src.graph.gate import Gate
        from src.retrieval.hibrida import Retriever
        from src.retrieval.reranking import CrossEncoderReranker

        return Pipeline(Retriever.load(CrossEncoderReranker()), None, gate=Gate.load()), str(exc)


# ------------------------------------------------------------------ resposta


def decision(state: dict) -> tuple[str, str]:
    """Titulo e tipo da caixa de decisao (success, warning ou error)."""
    d = state.get("decisao")
    if d == "responder":
        return "Resposta fundamentada nas avaliações", "success"
    if d == "erro_llm":
        return "O modelo de linguagem falhou; as evidências encontradas estão abaixo", "error"
    motivo = MOTIVOS.get(state.get("motivo") or "", "sem evidência suficiente")
    return f"Pergunta recusada: {motivo}", "warning"


def frequency_table(state: dict, top: int = 8) -> pd.DataFrame:
    freq = state.get("frequencia") or []
    if not freq:
        return pd.DataFrame(columns=["Tema", "Avaliações", "%"])
    df = pd.DataFrame(freq)[["tema", "avaliacoes", "pct"]].head(top)
    return df.rename(columns={"tema": "Tema", "avaliacoes": "Avaliações", "pct": "%"})


def cited_evidence(state: dict) -> list[dict]:
    """Evidencias citadas por alguma afirmacao, na ordem; todas quando a geracao falhou."""
    evidence = state.get("evidencias") or []
    if state.get("decisao") == "erro_llm":
        return evidence
    cited = {c for a in (state.get("resposta") or {}).get("afirmacoes", [])
             for c in a["evidencias"]}
    return [e for e in evidence if e["id"] in cited]


def score_label(nota: int | None) -> str:
    return f"nota {nota} de 5" if nota else ""


def retrieval_table(state: dict) -> pd.DataFrame:
    """Entradas recuperadas com o score final e a posicao em cada ramo da busca."""
    hits = state.get("hits") or []
    if not hits:
        return pd.DataFrame()
    return pd.DataFrame([{
        "texto": h["text"][:90], "score": round(h["score"], 3),
        "posição BM25": h.get("rank_lexical"), "posição vetorial": h.get("rank_vector"),
        "avaliações": h["n_reviews"]} for h in hits])


def stage_times(state: dict) -> dict[str, float]:
    t = state.get("tempos") or {}
    names = {"analisar_llm": "análise (LLM)", "recuperar": "recuperação",
             "gerar_llm": "geração (LLM)"}
    return {label: round(t[k], 2) for k, label in names.items() if k in t}


# ------------------------------------------------------------------ qualidade


def quality_summary() -> dict:
    """Numeros da avaliacao lidos dos artefatos versionados."""
    out: dict = {}
    if ABLATION.exists():
        out["ablacao"] = pd.read_csv(ABLATION)
    if TRACES.exists():
        tr = pd.read_parquet(TRACES)
        test = tr[tr["split"] == "teste"]
        esp, obt = test["tipo"] == "respondivel", test["decisao"] == "responder"
        out["recusa"] = (int((~esp & ~obt).sum()), int((~esp).sum()))
        out["respondidas"] = (int((esp & obt).sum()), int(esp.sum()))
        resp = tr[tr["decisao"] == "responder"]
        out["validade"] = float(resp["validade_citacoes"].mean())
        tempos = pd.DataFrame([json.loads(x) for x in resp["tempos"]])
        cols = [c for c in ("analisar_llm", "recuperar", "gerar_llm") if c in tempos]
        out["tempo_mediano"] = float(tempos[cols].sum(axis=1).median())
    if FAITHFULNESS.exists():
        out["fidelidade"] = float(pd.read_parquet(FAITHFULNESS)["sustentada"].mean())
    if THRESHOLD.exists():
        out["limiar"] = json.loads(THRESHOLD.read_text())["limiar"]
    if METRICS.exists():
        m = re.search(r"\| pipeline final \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|",
                      METRICS.read_text(encoding="utf-8"))
        if m:
            out["final"] = {"recall": float(m.group(1)), "precision": float(m.group(2)),
                            "mrr": float(m.group(3))}
    return out


def br(value: float, decimals: int = 2) -> str:
    """Numero com virgula decimal."""
    return f"{value:.{decimals}f}".replace(".", ",")


def ablation_view(abl: pd.DataFrame) -> pd.DataFrame:
    """Tabela da ablacao com nomes de leitura e virgula decimal."""
    out = pd.DataFrame({
        "Variante": abl["nome"],
        "recall@10": abl["recall@10"].map(lambda v: br(v, 3)),
        "precision@10": abl["precision@10"].map(lambda v: br(v, 3)),
        "MRR": abl["mrr"].map(lambda v: br(v, 3)),
        "Latência mediana (ms)": abl["latencia_mediana_ms"].map(lambda v: f"{v:.0f}"),
    })
    return out


# ------------------------------------------------------------------ auditoria


def load_audit(path: Path = AUDIT) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, dtype={"nota_manual": str})
    df["nota_manual"] = df["nota_manual"].astype(str).str.strip()
    return df


def save_label(df: pd.DataFrame, index: int, nota: int, path: Path = AUDIT) -> pd.DataFrame:
    if nota not in (0, 1, 2):
        raise ValueError("a nota vai de 0 a 2")
    out = df.copy()
    out.loc[index, "nota_manual"] = str(nota)
    out.to_csv(path, index=False)
    return out


def next_unlabeled(df: pd.DataFrame) -> int | None:
    pending = df.index[df["nota_manual"] == ""]
    return int(pending[0]) if len(pending) else None


def audit_agreement(df: pd.DataFrame) -> dict:
    """Concordancia da anotacao humana com os dois juizes, nos pares ja rotulados."""
    done = df[df["nota_manual"] != ""]
    if done.empty:
        return {"rotulados": 0}
    m = done["nota_manual"].astype(int).to_numpy()
    out = {"rotulados": len(done)}
    for col, name in [("nota_juiz", "juiz"), ("nota_segundo_juiz", "segundo_juiz")]:
        j = done[col].astype(int).to_numpy()
        out[f"concordancia_{name}"] = float(np.mean(m == j))
        out[f"kappa_{name}"] = cohen_kappa(m, j) if len(set(m) | set(j)) > 1 else float("nan")
    return out
