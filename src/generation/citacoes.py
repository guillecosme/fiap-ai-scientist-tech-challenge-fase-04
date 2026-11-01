"""Verificacao das citacoes por codigo (D11).

Toda afirmacao precisa citar ao menos uma evidencia que de fato foi entregue ao LLM. A
verificacao separa as afirmacoes validas das invalidas (sem citacao ou com id que nao
existe), mede a validade e devolve a resposta so com as validas: o gestor nunca ve uma
afirmacao sem evidencia rastreavel.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.generation.esquemas import Afirmacao, Resposta


@dataclass
class Verificacao:
    afirmacoes: int
    validas: int
    citacoes: int
    citacoes_validas: int
    removidas: list[str]

    @property
    def validade(self) -> float:
        """Fracao das afirmacoes com todas as citacoes validas."""
        return self.validas / self.afirmacoes if self.afirmacoes else 1.0

    @property
    def validade_citacoes(self) -> float:
        return self.citacoes_validas / self.citacoes if self.citacoes else 1.0


def verify(resposta: Resposta, evidence_ids: set[str]) -> tuple[Resposta, Verificacao]:
    keep: list[Afirmacao] = []
    removed: list[str] = []
    n_cit = n_ok = 0
    for a in resposta.afirmacoes:
        cited = [c.strip().upper() for c in a.evidencias]
        ok = [c for c in cited if c in evidence_ids]
        n_cit += len(cited)
        n_ok += len(ok)
        if cited and len(ok) == len(cited):
            keep.append(Afirmacao(texto=a.texto, evidencias=ok))
        else:
            removed.append(a.texto)
    checked = resposta.model_copy(update={"afirmacoes": keep})
    if resposta.suficiente and not keep:
        checked = checked.model_copy(update={"suficiente": False})
    return checked, Verificacao(len(resposta.afirmacoes), len(keep), n_cit, n_ok, removed)
