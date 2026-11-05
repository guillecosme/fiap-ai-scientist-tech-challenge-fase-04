"""Gate de evidencia (D12): o fluxo so gera resposta quando a busca trouxe evidencia.

Uso:
    python -m src.graph.gate coletar    # analise e busca de cada pergunta de avaliacao
    python -m src.graph.gate calibrar   # limiar nas perguntas de calibracao

A decisao usa tres sinais, nesta ordem:
1. nenhuma avaliacao com texto no recorte pedido: recusa "sem avaliacoes";
2. menos de `min_reviews` avaliacoes no recorte: recusa "poucas avaliacoes";
3. o maior score do cross-encoder entre os resultados abaixo do limiar: recusa
   "evidencia insuficiente".

O limiar e calibrado nas perguntas de calibracao do conjunto de avaliacao (notebook 06
e `reports/limiar_evidencia.csv`): e o valor que maximiza a acuracia balanceada entre
perguntas respondiveis (devem passar) e nao respondiveis (devem ser recusadas). As
perguntas de teste nao participam da escolha.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLD_FILE = Path("data/eval/limiar_evidencia.json")
FEATURES_FILE = Path("data/eval/gate_sinais.parquet")
CURVE_FILE = Path("reports/limiar_evidencia.csv")
MIN_REVIEWS = 3


def _score(h: dict) -> float:
    return h["score_rerank"] if h.get("score_rerank") is not None else h["score"]


@dataclass
class Gate:
    threshold: float = 0.0
    min_hits: int = 1
    min_reviews: int = MIN_REVIEWS

    @classmethod
    def load(cls, path: Path = THRESHOLD_FILE) -> Gate:
        if path.exists():
            data = json.loads(path.read_text())
            return cls(threshold=data["limiar"], min_reviews=data.get("min_reviews",
                                                                       MIN_REVIEWS))
        return cls()

    def decide(self, hits: list[dict], total_recorte: int) -> dict:
        scores = [_score(h) for h in hits]
        top = max(scores) if scores else None
        above = sum(sc >= self.threshold for sc in scores)
        if total_recorte == 0 or not hits:
            motivo = "sem_avaliacoes"
        elif total_recorte < self.min_reviews:
            motivo = "poucas_avaliacoes"
        elif above < self.min_hits:
            motivo = "evidencia_insuficiente"
        else:
            motivo = None
        return {"passa": motivo is None, "motivo": motivo, "top_score": top,
                "acima_do_limiar": int(above), "limiar": self.threshold}


def threshold_curve(top_scores: np.ndarray, answerable: np.ndarray) -> pd.DataFrame:
    """Para cada limiar candidato: taxa de passagem das respondiveis e de recusa das demais.

    top_scores: maior score de cada pergunta (NaN quando nao houve resultado, o que conta
    como recusa em qualquer limiar).
    """
    s = np.where(np.isnan(top_scores), -np.inf, top_scores)
    finite = np.unique(s[np.isfinite(s)])
    candidates = np.concatenate([[-np.inf], (finite[:-1] + finite[1:]) / 2, [np.inf]]) \
        if len(finite) > 1 else np.array([-np.inf, np.inf])
    rows = []
    for t in candidates:
        passes = s >= t
        tpr = passes[answerable].mean() if answerable.any() else np.nan
        tnr = (~passes[~answerable]).mean() if (~answerable).any() else np.nan
        rows.append({"limiar": t, "passa_respondiveis": tpr, "recusa_nao_respondiveis": tnr,
                     "acuracia_balanceada": (tpr + tnr) / 2})
    return pd.DataFrame(rows)


def calibrate(top_scores: np.ndarray, answerable: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Limiar de maior acuracia balanceada; empate resolvido pelo limiar mais baixo
    (prefere responder a recusar quando os dois erram igual)."""
    curve = threshold_curve(np.asarray(top_scores, dtype=float), np.asarray(answerable, bool))
    finite = curve[np.isfinite(curve["limiar"])]
    best = finite.sort_values(["acuracia_balanceada", "limiar"],
                              ascending=[False, True]).iloc[0]
    return float(best["limiar"]), curve


def save_threshold(threshold: float, path: Path = THRESHOLD_FILE, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"limiar": threshold, "min_reviews": MIN_REVIEWS, **extra},
                               indent=2) + "\n")


def collect_features(pipeline, questions: pd.DataFrame) -> pd.DataFrame:
    """Analise e busca de cada pergunta, sem gerar resposta: o que o gate ve."""
    rows = []
    for _, q in questions.iterrows():
        s = {"pergunta": q["pergunta"], "tempos": {}, "erros": []}
        s.update(pipeline.analisar(s))
        row = {"question_id": q["id"], "tipo": q["tipo"], "split": q["split"],
               "escopo": s["analise"]["escopo"], "fonte_analise": s["fonte_analise"],
               "filtros": json.dumps(s["filtros"], ensure_ascii=False),
               "consulta": s["analise"]["consulta"], "frequencia": s["analise"]["frequencia"],
               "total_recorte": None, "top_score": np.nan, "n_hits": 0}
        if s["analise"]["escopo"] == "dentro":
            s.update(pipeline.recuperar(s))
            scores = [_score(h) for h in s["hits"]]
            row.update({"total_recorte": s["total_recorte"], "n_hits": len(scores),
                        "top_score": max(scores) if scores else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("etapa", choices=["coletar", "calibrar"])
    args = parser.parse_args()
    if args.etapa == "coletar":
        from src.evaluation.julgamento import load_questions
        from src.graph.fluxo import Pipeline

        df = collect_features(Pipeline.load(), load_questions())
        FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(FEATURES_FILE, index=False)
        print(df.groupby(["tipo", "escopo"]).size().to_string())
    else:
        df = pd.read_parquet(FEATURES_FILE)
        # fora do escopo ja e recusado pela analise; o gate decide o resto
        cal = df[(df["split"] == "calibracao") & (df["escopo"] == "dentro")]
        threshold, curve = calibrate(cal["top_score"].to_numpy(),
                                     (cal["tipo"] == "respondivel").to_numpy())
        CURVE_FILE.parent.mkdir(parents=True, exist_ok=True)
        curve.to_csv(CURVE_FILE, index=False)
        save_threshold(round(threshold, 4), perguntas_calibracao=int(len(cal)))
        print("limiar:", round(threshold, 4))


if __name__ == "__main__":
    main()
