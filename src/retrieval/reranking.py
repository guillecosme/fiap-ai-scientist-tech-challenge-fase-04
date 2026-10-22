"""Re-ranking com cross-encoder multilingue (D08).

O bi-encoder da busca vetorial codifica pergunta e avaliacao separadas, o que e rapido
mas perde a interacao entre as duas (a negacao, por exemplo). O cross-encoder le o par
inteiro e da um score de relevancia; e caro demais para a base toda, entao so reordena
os 50 candidatos da busca hibrida.

Modelo: mmarco-mMiniLMv2-L12-H384-v1, treinado no mMARCO (MS MARCO traduzido por maquina
para varias linguas, incluindo portugues).
"""

from __future__ import annotations

import os
from functools import cache

import numpy as np

MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


@cache
def _load(model_name: str = MODEL_NAME):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device="cpu", max_length=256)


class CrossEncoderReranker:
    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size

    def score(self, question: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        model = _load(self.model_name)
        return np.asarray(model.predict([(question, t) for t in texts],
                                        batch_size=self.batch_size, show_progress_bar=False))
