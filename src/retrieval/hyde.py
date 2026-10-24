"""HyDE: busca vetorial com o embedding de uma avaliacao hipotetica.

A pergunta do gestor ("quais os principais motivos de insatisfacao?") nao se parece com
uma avaliacao de cliente ("comprei e nao chegou ate hoje"). O HyDE pede ao LLM uma
avaliacao curta que responderia a pergunta e usa o embedding dela na busca vetorial; o
BM25 continua com a pergunta. O texto gerado nunca vira evidencia: so orienta a busca.
"""

from __future__ import annotations

import numpy as np

from src.indexing.vetorial import embed

PROMPT = (
    "Escreva duas avaliações curtas e diferentes, de no máximo 25 palavras cada, como "
    "clientes brasileiros escreveriam no site de um e-commerce, que serviriam de exemplo "
    "para responder à pergunta abaixo. Escreva só as avaliações, uma por linha, sem "
    "numeração.\n\nPergunta: {question}"
)


def hyde_text(llm, question: str) -> str:
    return llm.invoke(PROMPT.format(question=question)).strip()


def hyde_vector(text: str) -> np.ndarray:
    """Media normalizada dos embeddings das linhas geradas."""
    lines = [line.strip() for line in text.splitlines() if line.strip()] or [text]
    v = embed(lines).mean(axis=0)
    return v / np.linalg.norm(v)
