"""Variantes de busca, pool de candidatos, julgamento e amostra de auditoria.

As variantes sao as mesmas da ablacao do notebook 05. O pool junta os 10 primeiros de
cada variante para cada pergunta; o julgamento cobre o pool inteiro, entao todas as
variantes sao medidas contra os mesmos julgamentos.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from src.evaluation.julgamento import AUDIT, JUDGMENTS, POOL, POOL_DEPTH, judge, load_questions
from src.retrieval.filtros import rule_based_filters

HYDE = POOL.parent / "hyde.parquet"
SECOND_JUDGE = "gemma-4-26b-a4b-it"
AUDIT_QUESTIONS = 30
PAIRS_PER_QUESTION = 4

# nome: (modo, re-ranking, filtro, HyDE)
VARIANTS = {
    "bm25": ("bm25", False, None, False),
    "vetorial": ("vector", False, None, False),
    "hibrida": ("hybrid", False, None, False),
    "hibrida_rerank": ("hybrid", True, None, False),
    "hibrida_rerank_filtros": ("hybrid", True, "regras", False),
    "hibrida_rerank_filtros_hyde": ("hybrid", True, "regras", True),
}


def fast_mode(questions: pd.DataFrame) -> pd.DataFrame:
    """TC_RAPIDO=1: uma pergunta de cada cinco, para iterar sem gastar cota."""
    if os.environ.get("TC_RAPIDO") == "1":
        return questions.iloc[::5]
    return questions


def workers() -> int:
    """Chamadas simultaneas ao LLM (TC_LLM_PARALELO, padrao 1): um nivel de paralelismo."""
    return max(1, int(os.environ.get("TC_LLM_PARALELO", "1")))


def load_hyde() -> dict[str, str]:
    if not HYDE.exists():
        return {}
    df = pd.read_parquet(HYDE)
    return dict(zip(df["question_id"], df["hyde"], strict=True))


def build_hyde(questions: pd.DataFrame) -> dict[str, str]:
    from src.generation.provedor import LLM
    from src.retrieval.hyde import hyde_text

    texts = load_hyde()
    llm = LLM("chat")
    todo = [(q["id"], q["pergunta"]) for _, q in questions.iterrows() if q["id"] not in texts]
    with ThreadPoolExecutor(workers()) as ex:
        for qid, text in zip([t[0] for t in todo],
                             ex.map(lambda t: hyde_text(llm, t[1]), todo), strict=True):
            texts[qid] = text
    HYDE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"question_id": list(texts), "hyde": list(texts.values())}).to_parquet(
        HYDE, index=False)
    print("HyDE:", llm.usage())
    return texts


def run_variant(retriever, question: str, variant: str, hyde: str | None,
                k: int = POOL_DEPTH) -> list[str]:
    from src.retrieval.hyde import hyde_vector

    mode, rerank, filt, use_hyde = VARIANTS[variant]
    filters = rule_based_filters(question) if filt == "regras" else None
    qv = hyde_vector(hyde) if use_hyde and hyde else None
    hits = retriever.retrieve(question, filters, mode=mode, k=k, rerank=rerank,
                              mmr_lambda=None, query_vector=qv)
    return [h.doc_id for h in hits]


def build_pool() -> pd.DataFrame:
    from src.retrieval.hibrida import Retriever
    from src.retrieval.reranking import CrossEncoderReranker

    questions = fast_mode(load_questions())
    hyde = build_hyde(questions)
    retriever = Retriever.load(CrossEncoderReranker())
    rows = []
    t0 = time.perf_counter()
    for _, q in questions.iterrows():
        for variant in VARIANTS:
            for rank, doc_id in enumerate(run_variant(retriever, q["pergunta"], variant,
                                                      hyde.get(q["id"])), 1):
                rows.append({"question_id": q["id"], "variant": variant, "rank": rank,
                             "doc_id": doc_id})
    pool = pd.DataFrame(rows)
    pool.to_parquet(POOL, index=False)
    per_q = pool.groupby("question_id")["doc_id"].nunique()
    print(f"pool: {len(questions)} perguntas, {per_q.sum()} pares a julgar "
          f"(mediana {per_q.median():.0f} por pergunta), {time.perf_counter() - t0:.0f} s")
    return pool


def judge_pool(model_override: str | None = None, out=JUDGMENTS) -> pd.DataFrame:
    from src.generation.provedor import LLM

    questions = load_questions().set_index("id")
    pool = pd.read_parquet(POOL)
    docs = pd.read_parquet("data/processed/documentos.parquet")
    texts = docs[docs["is_representative"]].set_index("doc_id")["text_clean"]
    llm = LLM("judge", model_override=model_override)
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame(
        columns=["question_id", "doc_id", "nota", "juiz"])
    rows = [done]
    todo = []
    for qid, g in pool.groupby("question_id", sort=False):
        ids = sorted(set(g["doc_id"]) - set(done.loc[done["question_id"] == qid, "doc_id"]))
        if ids:
            todo.append((qid, ids))

    def run(task):
        qid, ids = task
        grades = judge(llm, questions.loc[qid, "pergunta"], texts.loc[ids].tolist())
        return pd.DataFrame({"question_id": qid, "doc_id": ids, "nota": grades,
                             "juiz": llm.model})

    with ThreadPoolExecutor(workers()) as ex:
        for part in ex.map(run, todo):
            rows.append(part)
            pd.concat(rows, ignore_index=True).to_parquet(out, index=False)
    result = pd.concat(rows, ignore_index=True)
    result["nota"] = result["nota"].astype(int)
    result.to_parquet(out, index=False)
    print("julgamento:", llm.usage())
    return result


def build_audit() -> pd.DataFrame:
    """Amostra para auditoria: 30 perguntas respondiveis sorteadas e 4 pares de cada,
    priorizando notas diferentes do juiz. O segundo juiz julga a amostra (uma chamada por
    pergunta) e a coluna nota_manual fica vazia para o rotulo humano; a concordancia
    com ela e calculada no notebook 05 quando estiver preenchida."""
    from src.generation.provedor import LLM

    questions = load_questions().set_index("id")
    j = pd.read_parquet(JUDGMENTS)
    j = j[j["question_id"].str.startswith("q")]
    rng = np.random.default_rng(42)
    chosen = rng.choice(sorted(j["question_id"].unique()), size=AUDIT_QUESTIONS, replace=False)
    parts = []
    for qid in sorted(chosen):
        g = j[j["question_id"] == qid].sample(frac=1, random_state=42)
        # um de cada nota primeiro, depois o que faltar
        pick = g.drop_duplicates("nota")
        pick = pd.concat([pick, g.drop(pick.index)]).head(PAIRS_PER_QUESTION)
        parts.append(pick)
    sample = pd.concat(parts, ignore_index=True)
    docs = pd.read_parquet("data/processed/documentos.parquet")
    texts = docs[docs["is_representative"]].set_index("doc_id")["text_clean"]
    second = LLM("judge", model_override=SECOND_JUDGE)
    second_grades = {}
    for qid, g in sample.groupby("question_id"):
        for d, n in zip(g["doc_id"], judge(second, questions.loc[qid, "pergunta"],
                                           texts.loc[g["doc_id"]].tolist()), strict=True):
            second_grades[(qid, d)] = n
    audit = pd.DataFrame({
        "question_id": sample["question_id"],
        "pergunta": sample["question_id"].map(questions["pergunta"]),
        "doc_id": sample["doc_id"],
        "texto": sample["doc_id"].map(texts),
        "nota_juiz": sample["nota"].astype(int),
        "nota_segundo_juiz": [second_grades[(q, d)] for q, d in
                              zip(sample["question_id"], sample["doc_id"], strict=True)],
        "nota_manual": "",
    })
    audit.to_csv(AUDIT, index=False)
    print("segundo juiz:", second.usage())
    return audit
