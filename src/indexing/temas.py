"""Mapa de temas das avaliacoes com BERTopic, separado por polaridade.

Uso:
    python -m src.indexing.temas    # grava data/index/temas.parquet e reports/temas.csv

Dois modelos, um por polaridade (D23): textos cuja nota media e ate 3 (problemas) e
textos com nota media acima de 3 (elogios). Num modelo unico, o mesmo assunto juntava
elogio e reclamacao no mesmo topico ("nao foi entregue" e "chegou antes do prazo"),
porque o embedding pesa pouco a polaridade (notebook 03).

Cada modelo: UMAP (5 dimensoes, cosseno) sobre os embeddings ja calculados, HDBSCAN,
representacao por c-TF-IDF com as stopwords do projeto, reducao hierarquica do numero
de topicos e realocacao dos textos sem topico pelo embedding mais proximo.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.indexing.vetorial import load_embeddings
from src.preprocessing.deduplicacao import unique_docs
from src.preprocessing.tokenizacao import stopwords

DOCS = Path("data/processed/documentos.parquet")
THEMES = Path("data/index/temas.parquet")
REPORT = Path("reports/temas.csv")
SEED = 42

# parametros por lado: tamanho minimo do grupo no HDBSCAN e numero final de topicos
SIDES = {
    "problema": {"min_cluster_size": 40, "nr_topics": 20},
    "elogio": {"min_cluster_size": 100, "nr_topics": 15},
}
OUTLIER_THRESHOLD = 0.5


def side_of(mean_score: np.ndarray) -> np.ndarray:
    return np.where(mean_score <= 3, "problema", "elogio")


def _vectorizer():
    from sklearn.feature_extraction.text import CountVectorizer

    return CountVectorizer(stop_words=sorted(stopwords()), ngram_range=(1, 2), min_df=5)


def fit_side(texts: list[str], embeddings: np.ndarray, min_cluster_size: int, nr_topics: int):
    """Ajusta o BERTopic de um lado e devolve (modelo, topico de cada texto)."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    vec = _vectorizer()
    model = BERTopic(
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                        random_state=SEED),
        hdbscan_model=HDBSCAN(min_cluster_size=min_cluster_size, min_samples=10,
                              prediction_data=True),
        vectorizer_model=vec, nr_topics=nr_topics, calculate_probabilities=False)
    topics, _ = model.fit_transform(texts, embeddings=embeddings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        topics = model.reduce_outliers(texts, topics, strategy="embeddings",
                                       embeddings=embeddings, threshold=OUTLIER_THRESHOLD)
        model.update_topics(texts, topics=topics, vectorizer_model=vec)
    return model, np.asarray(topics)


def fit_topics(docs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Topico fino de cada texto distinto, por lado, e os modelos ajustados."""
    uniq = unique_docs(docs)
    ids, vectors = load_embeddings()
    if ids != uniq["doc_id"].tolist():
        raise ValueError("embeddings fora de ordem: rode make index de novo")
    mean_score = docs.groupby("doc_id")["review_score"].mean().reindex(ids).to_numpy()
    side = side_of(mean_score)
    texts = uniq["text_norm"].to_numpy()
    parts, models = [], {}
    for name, params in SIDES.items():
        sel = side == name
        model, topics = fit_side(texts[sel].tolist(), vectors[sel], **params)
        models[name] = model
        parts.append(pd.DataFrame({"doc_id": np.asarray(ids)[sel], "polaridade": name,
                                   "topico": topics}))
    return pd.concat(parts, ignore_index=True), models


def topic_words(model, topic: int, n: int = 10) -> list[str]:
    return [w for w, _ in (model.get_topic(topic) or [])[:n]]


def lda_topics(texts: list[list[str]], n_topics: int, passes: int = 10):
    """LDA (gensim) como linha de base: devolve as 10 palavras de cada topico."""
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel

    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(t) for t in texts]
    lda = LdaModel(corpus, id2word=dictionary, num_topics=n_topics, passes=passes,
                   random_state=SEED, alpha="auto", eta="auto")
    return [[w for w, _ in lda.show_topic(k, topn=10)] for k in range(n_topics)], lda


def coherence_cv(topics: list[list[str]], texts: list[list[str]]) -> list[float]:
    """Coerencia c_v de cada topico (palavras unicas, sem bigramas)."""
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel

    dictionary = Dictionary(texts)
    clean = [[w for w in t if w in dictionary.token2id] for t in topics]
    clean = [t for t in clean if len(t) >= 2]
    cm = CoherenceModel(topics=clean, texts=texts, dictionary=dictionary, coherence="c_v",
                        processes=1)
    return cm.get_coherence_per_topic()


# topico fino -> (tema nomeado, palavra que precisa estar entre as 10 do topico). A
# palavra protege contra uma numeracao diferente em outra maquina: se nao bater, o
# topico vira "Outros" e o aviso aparece no log.
THEME_MAP = {
    "problema": {
        -1: ("Outros problemas", None),
        0: ("Pedido não recebido", "recebi"),
        6: ("Pedido não recebido", "recebi"),
        8: ("Pedido não recebido", "recebi"),
        11: ("Pedido não recebido", "aguardando"),
        12: ("Pedido não recebido", "aguardando"),
        1: ("Pedido incompleto, faltando itens", "apenas"),
        9: ("Pedido incompleto, faltando itens", "faltando"),
        14: ("Pedido incompleto, faltando itens", "cortina"),
        2: ("Defeito, produto errado ou má qualidade", "qualidade"),
        16: ("Defeito, produto errado ou má qualidade", "defeito"),
        18: ("Defeito, produto errado ou má qualidade", "qualidade"),
        10: ("Defeito, produto errado ou má qualidade", "relogio"),
        5: ("Acessórios eletrônicos que não funcionam", "celular"),
        4: ("Cor, tamanho ou aparência diferente do anúncio", "cor"),
        13: ("Cor, tamanho ou aparência diferente do anúncio", "tamanho"),
        7: ("Atraso e prazo de entrega longo", "prazo"),
        17: ("Atraso e prazo de entrega longo", "natal"),
        15: ("Descaso da loja com o cliente", "cliente"),
        3: ("Satisfeito, com ressalvas", "bom"),
    },
    "elogio": {
        -1: ("Outros elogios", None),
        0: ("Produto e entrega conforme o combinado", "produto"),
        1: ("Entrega antes ou dentro do prazo", "antes"),
        6: ("Entrega antes ou dentro do prazo", "prazo"),
        3: ("Entrega rápida", "rapida"),
        2: ("Elogio curto e genérico", "bom"),
        12: ("Elogio curto e genérico", "10"),
        8: ("Beleza e acabamento do produto", "relogio"),
        10: ("Beleza e acabamento do produto", "mochila"),
        13: ("Beleza e acabamento do produto", "cortina"),
        7: ("Atendimento, site e confiança na loja", "atendimento"),
        11: ("Atendimento, site e confiança na loja", "site"),
        4: ("Satisfeito, com ressalvas", "ainda"),
        5: ("Satisfeito, com ressalvas", "correios"),
        9: ("Satisfeito, com ressalvas", "cor"),
    },
}


def name_topics(assign: pd.DataFrame, models: dict) -> pd.DataFrame:
    out = assign.copy()
    names = []
    for side, topic in zip(out["polaridade"], out["topico"], strict=True):
        theme, word = THEME_MAP[side].get(topic, (None, None))
        names.append(theme or f"Outros {'problemas' if side == 'problema' else 'elogios'}")
    out["tema"] = names
    for side, mapping in THEME_MAP.items():
        for topic, (_theme, word) in mapping.items():
            if word and word not in topic_words(models[side], topic):
                warnings.warn(f"{side} {topic}: '{word}' fora das palavras do topico; "
                              f"confira o THEME_MAP", stacklevel=2)
                out.loc[(out["polaridade"] == side) & (out["topico"] == topic),
                        "tema"] = f"Outros {'problemas' if side == 'problema' else 'elogios'}"
    return out


def theme_report(themes: pd.DataFrame, docs: pd.DataFrame, models: dict) -> pd.DataFrame:
    rev = docs.merge(themes, on="doc_id")
    rev["nao_recebeu"] = rev["delivery_status"].isin(["nao entregue", "cancelada"]) | rev[
        "answered_before_delivery"]
    words = {}
    for (side, theme), g in themes.groupby(["polaridade", "tema"]):
        topics = g["topico"].value_counts().index
        ws = []
        for t in topics:
            ws += [w for w in topic_words(models[side], t, 6) if w not in ws]
        words[(side, theme)] = ", ".join(ws[:10])
    rep = rev.groupby(["polaridade", "tema"]).agg(
        avaliacoes=("review_id", "size"), textos=("doc_id", "nunique"),
        nota_media=("review_score", "mean"),
        pct_atrasada=("delivery_status", lambda s: 100 * (s == "atrasada").mean()),
        pct_nao_recebeu=("nao_recebeu", lambda s: 100 * s.mean()),
    ).round(2).reset_index()
    rep["pct_do_lado"] = (100 * rep["avaliacoes"] / rep.groupby("polaridade")[
        "avaliacoes"].transform("sum")).round(1)
    rep["palavras"] = [words[(s, t)] for s, t in zip(rep["polaridade"], rep["tema"],
                                                     strict=True)]
    return rep.sort_values(["polaridade", "avaliacoes"], ascending=[False, False])


def load_themes(path: Path = THEMES) -> pd.DataFrame:
    return pd.read_parquet(path)


def theme_frequency(docs: pd.DataFrame, themes: pd.DataFrame,
                    mask: pd.Series | None = None) -> pd.DataFrame:
    """Contagem de avaliacoes por tema dentro de um recorte (D10).

    A frequencia sai da base inteira filtrada, nao do top-k recuperado: e o que permite
    responder "quais os principais problemas" com numeros.
    """
    sel = docs if mask is None else docs[mask]
    rev = sel[["review_id", "doc_id"]].merge(themes[["doc_id", "polaridade", "tema"]],
                                             on="doc_id")
    freq = rev.groupby(["polaridade", "tema"]).size().rename("avaliacoes").reset_index()
    freq["pct"] = (100 * freq["avaliacoes"] / len(rev)).round(1) if len(rev) else 0.0
    return freq.sort_values("avaliacoes", ascending=False).reset_index(drop=True)


def main() -> None:
    docs = pd.read_parquet(DOCS)
    assign, models = fit_topics(docs)
    themes = name_topics(assign, models)
    THEMES.parent.mkdir(parents=True, exist_ok=True)
    themes.to_parquet(THEMES, index=False)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report = theme_report(themes, docs, models)
    report.to_csv(REPORT, index=False)
    print(report[["polaridade", "tema", "avaliacoes", "pct_do_lado", "nota_media"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
