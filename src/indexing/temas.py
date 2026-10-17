"""Mapa de temas das avaliacoes com BERTopic, separado por polaridade.

Uso:
    python -m src.indexing.temas    # grava data/index/temas.parquet

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




def main() -> None:
    docs = pd.read_parquet(DOCS)
    assign, models = fit_topics(docs)
    THEMES.parent.mkdir(parents=True, exist_ok=True)
    assign.to_parquet(THEMES, index=False)
    print(assign.groupby("polaridade")["topico"].nunique().to_string())


if __name__ == "__main__":
    main()
