"""Limpeza e normalizacao dos textos das avaliacoes.

Tres versoes do mesmo texto, cada uma com um uso (decisao D03):
- original: titulo e comentario como o cliente escreveu, so para exibicao da evidencia;
- limpa (`clean_text`): quebras de linha, emojis e pontuacao repetida tratados, com
  acentos e maiusculas preservados. E o texto que vai para o modelo de embeddings e
  para o contexto do LLM;
- normalizada (`normalize`): minusculas, sem acento, so letras e digitos. E a chave de
  deduplicacao e a base da tokenizacao do indice lexical.
"""

from __future__ import annotations

import re
import unicodedata

# emojis, pictogramas, setas, simbolos diversos e seletores de variacao
_SYMBOLS = re.compile(
    "[\U0001F000-\U0001FAFF\U00002190-\U000021FF\U00002300-\U000027BF"
    "\U00002B00-\U00002BFF\U0000FE0F\U0000200D]+"
)
_REPEATED_PUNCT = re.compile(r"([!?.,;:*\-])\1+")
_SPACES = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_REPEATED_CHAR = re.compile(r"([a-z])\1{2,}")


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def clean_text(text: str | None) -> str:
    """Texto limpo para embeddings e contexto do LLM."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _SYMBOLS.sub(" ", text)
    text = _REPEATED_PUNCT.sub(r"\1", text)
    return _SPACES.sub(" ", text).strip()


def normalize(text: str | None) -> str:
    """Forma normalizada: minusculas, sem acento, so [a-z0-9] separados por espaco.

    Letras repetidas tres ou mais vezes viram uma ("muitooo" -> "muito",
    "kkkk" -> "k"), para que variacoes de enfase caiam na mesma chave.
    """
    text = strip_accents(clean_text(text).lower())
    text = _NON_ALNUM.sub(" ", text)
    text = _REPEATED_CHAR.sub(r"\1", text)
    return _SPACES.sub(" ", text).strip()


def has_content(normalized: str) -> bool:
    """Ha alguma letra: texto so de pontuacao, numero ou emoji nao vai para o indice."""
    return bool(re.search(r"[a-z]", normalized))


def join_title_comment(title: str | None, comment: str | None) -> str:
    """Titulo e comentario limpos em um texto so; o titulo vira a primeira frase."""
    title, comment = clean_text(title), clean_text(comment)
    if title and comment:
        sep = " " if title[-1] in ".!?" else ". "
        return f"{title}{sep}{comment}"
    return title or comment
