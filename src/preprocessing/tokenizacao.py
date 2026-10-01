"""Tokenizacao para o indice lexical (BM25): stopwords do NLTK e stemming RSLP.

Passos, na ordem:
1. texto limpo em minusculas, quebrado em palavras (letras e digitos);
2. remocao das stopwords do NLTK para portugues, comparadas sem acento, exceto as
   negacoes "nao", "nem" e "sem": em avaliacoes elas mudam o sentido ("nao recebi"
   e "recebi" nao podem virar a mesma consulta);
3. stemming RSLP sobre a palavra com acento (as regras do RSLP usam acentos) e remocao
   do acento depois, para que "otimo" e "ótimo" caiam no mesmo radical.

A mesma funcao tokeniza documentos e perguntas, o que garante o mesmo vocabulario.
"""

from __future__ import annotations

import re
from functools import cache

import nltk

from src.preprocessing.limpeza import clean_text, strip_accents

KEEP_NEGATIONS = {"nao", "nem", "sem"}
_WORD = re.compile(r"[^\W_]+")
_REPEATED_CHAR = re.compile(r"([a-z])\1{2,}")


def ensure_nltk_data() -> None:
    """Baixa as stopwords e o RSLP do NLTK na primeira execucao."""
    for resource, package in [("corpora/stopwords", "stopwords"), ("stemmers/rslp", "rslp")]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)


@cache
def stopwords() -> frozenset[str]:
    ensure_nltk_data()
    words = {strip_accents(w) for w in nltk.corpus.stopwords.words("portuguese")}
    return frozenset(words - KEEP_NEGATIONS)


@cache
def _stemmer() -> nltk.stem.RSLPStemmer:
    ensure_nltk_data()
    return nltk.stem.RSLPStemmer()


@cache
def stem(word: str) -> str:
    return strip_accents(_stemmer().stem(word))


def tokenize(text: str | None) -> list[str]:
    """Radicais das palavras do texto, sem stopwords; usado em documentos e perguntas."""
    tokens = []
    for word in _WORD.findall(clean_text(text).lower()):
        plain = _REPEATED_CHAR.sub(r"\1", strip_accents(word))
        if plain in stopwords() or (len(plain) < 2 and not plain.isdigit()):
            continue
        tokens.append(stem(word if plain == strip_accents(word) else plain))
    return tokens
