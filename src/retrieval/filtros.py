"""Filtros por metadado aplicados antes da busca (D09).

`Filters` e o contrato entre a analise da pergunta e a recuperacao: o LLM preenche esse
schema com saida estruturada e, se falhar, `rule_based_filters` preenche por regras. Os
filtros valem por avaliacao (D04): uma entrada do indice concorre ao top-k quando pelo
menos uma das suas avaliacoes passa, e a contagem exibida e a das que passaram.
"""

from __future__ import annotations

import re
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.preprocessing.limpeza import normalize
from src.visualization.rotulos import CATEGORY_LABELS

DeliveryStatus = Literal["no prazo", "atrasada", "nao entregue", "cancelada"]
Region = Literal["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]

STATES = ["AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT", "PA",
          "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO"]


class Filters(BaseModel):
    """Recorte da base pedido pela pergunta. Campo vazio significa sem restricao."""

    score_min: int | None = Field(None, ge=1, le=5, description="nota minima (1 a 5)")
    score_max: int | None = Field(None, ge=1, le=5, description="nota maxima (1 a 5)")
    late: bool | None = Field(
        None, description="True: so pedidos entregues com atraso; False: so entregues no prazo")
    not_received: bool | None = Field(
        None, description="True: pedido nao entregue, cancelado ou cliente respondeu antes de "
        "receber (D19)")
    delivery_status: list[DeliveryStatus] | None = Field(
        None, description="situacoes de entrega aceitas")
    categories: list[str] | None = Field(
        None, description="categorias de produto (nome interno da Olist, ex.: cama_mesa_banho)")
    period_start: str | None = Field(None, description="mes inicial da compra, AAAA-MM")
    period_end: str | None = Field(None, description="mes final da compra, AAAA-MM")
    states: list[str] | None = Field(None, description="UFs do cliente, ex.: ['SP', 'RJ']")
    regions: list[Region] | None = Field(None, description="regioes do cliente")

    def is_empty(self) -> bool:
        return all(v in (None, []) for v in self.model_dump().values())

    def describe(self) -> str:
        """Descricao curta em portugues, para o rastro e para a resposta."""
        parts = []
        if self.score_min or self.score_max:
            lo, hi = self.score_min or 1, self.score_max or 5
            parts.append(f"nota de {lo} a {hi}")
        if self.late is True:
            parts.append("entregues com atraso")
        if self.late is False:
            parts.append("entregues no prazo")
        if self.not_received:
            parts.append("pedido nao recebido")
        if self.delivery_status:
            parts.append("entrega: " + ", ".join(self.delivery_status))
        if self.categories:
            parts.append("categoria: " + ", ".join(self.categories))
        if self.period_start or self.period_end:
            parts.append(f"compra de {self.period_start or 'inicio'} a {self.period_end or 'fim'}")
        if self.states:
            parts.append("UF: " + ", ".join(self.states))
        if self.regions:
            parts.append("regiao: " + ", ".join(self.regions))
        return "; ".join(parts) or "sem filtro"


def review_mask(docs: pd.DataFrame, f: Filters) -> pd.Series:
    """Avaliacoes (linhas de documentos) que passam no filtro."""
    m = pd.Series(True, index=docs.index)
    if f.score_min is not None:
        m &= docs["review_score"] >= f.score_min
    if f.score_max is not None:
        m &= docs["review_score"] <= f.score_max
    if f.late is True:
        m &= docs["delivery_status"].eq("atrasada")
    if f.late is False:
        m &= docs["delivery_status"].eq("no prazo")
    if f.not_received:
        m &= docs["delivery_status"].isin(["nao entregue", "cancelada"]) | docs[
            "answered_before_delivery"]
    if f.delivery_status:
        m &= docs["delivery_status"].isin(f.delivery_status)
    if f.categories:
        wanted = set(f.categories)
        m &= docs["categories"].map(lambda cats: bool(wanted.intersection(cats)))
    if f.period_start:
        m &= docs["purchase_month"] >= f.period_start
    if f.period_end:
        m &= docs["purchase_month"] <= f.period_end
    if f.states:
        m &= docs["customer_state"].isin(f.states)
    if f.regions:
        m &= docs["customer_region"].isin(f.regions)
    return m.fillna(False).astype(bool)


def entry_counts(docs: pd.DataFrame, mask: pd.Series, doc_ids: list[str]) -> np.ndarray:
    """Quantas avaliacoes de cada entrada do indice passaram no filtro (ordem de doc_ids)."""
    counts = docs.loc[mask, "doc_id"].value_counts()
    return counts.reindex(doc_ids, fill_value=0).to_numpy()


# ---------------------------------------------------------------- regras (fallback)

_MONTHS = {"janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4, "maio": 5, "junho": 6,
           "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11,
           "dezembro": 12}
_STATE_NAMES = {
    "acre": "AC", "alagoas": "AL", "amazonas": "AM", "amapa": "AP", "bahia": "BA",
    "ceara": "CE", "distrito federal": "DF", "espirito santo": "ES", "goias": "GO",
    "maranhao": "MA", "minas gerais": "MG", "mato grosso do sul": "MS", "mato grosso": "MT",
    "paraiba": "PB", "pernambuco": "PE", "piaui": "PI", "parana": "PR",
    "rio de janeiro": "RJ", "rio grande do norte": "RN", "rondonia": "RO", "roraima": "RR",
    "rio grande do sul": "RS", "santa catarina": "SC", "sergipe": "SE", "sao paulo": "SP",
    "tocantins": "TO",
}
_REGIONS = {"norte": "Norte", "nordeste": "Nordeste", "centro oeste": "Centro-Oeste",
            "sudeste": "Sudeste", "sul": "Sul"}

# termos de categoria escritos por um gestor -> categorias da Olist
_CATEGORY_TERMS = {
    "cama mesa e banho": ["cama_mesa_banho"], "cama": ["cama_mesa_banho"],
    "beleza": ["beleza_saude", "perfumaria"], "perfume": ["perfumaria"],
    "esporte": ["esporte_lazer", "fashion_esporte"], "informatica": ["informatica_acessorios",
                                                                      "pcs", "pc_gamer"],
    "computador": ["informatica_acessorios", "pcs", "pc_gamer"],
    "moveis": ["moveis_decoracao", "moveis_escritorio", "moveis_sala", "moveis_quarto",
               "moveis_cozinha_area_de_servico_jantar_e_jardim", "moveis_colchao_e_estofado"],
    "decoracao": ["moveis_decoracao"], "utilidades domesticas": ["utilidades_domesticas"],
    "relogio": ["relogios_presentes"], "presente": ["relogios_presentes"],
    "telefonia": ["telefonia"], "telefone fixo": ["telefonia_fixa"], "celular": ["telefonia"],
    "automotivo": ["automotivo"], "carro": ["automotivo"], "brinquedo": ["brinquedos"],
    "ferramenta": ["ferramentas_jardim", "construcao_ferramentas_construcao",
                   "construcao_ferramentas_ferramentas", "construcao_ferramentas_jardim",
                   "construcao_ferramentas_seguranca", "construcao_ferramentas_iluminacao"],
    "jardim": ["ferramentas_jardim", "construcao_ferramentas_jardim"],
    "bebe": ["bebes", "fraldas_higiene"], "eletronico": ["eletronicos"],
    "eletrodomestico": ["eletrodomesticos", "eletrodomesticos_2", "eletroportateis"],
    "papelaria": ["papelaria"], "pet": ["pet_shop"], "games": ["consoles_games"],
    "videogame": ["consoles_games"], "mala": ["malas_acessorios"], "bolsa":
    ["fashion_bolsas_e_acessorios"], "calcado": ["fashion_calcados"], "sapato":
    ["fashion_calcados"], "roupa": ["fashion_roupa_feminina", "fashion_roupa_masculina",
                                    "fashion_roupa_infanto_juvenil",
                                    "fashion_underwear_e_moda_praia"],
    "moda": ["fashion_bolsas_e_acessorios", "fashion_calcados", "fashion_esporte",
             "fashion_roupa_feminina", "fashion_roupa_masculina",
             "fashion_roupa_infanto_juvenil", "fashion_underwear_e_moda_praia"],
    "livro": ["livros_interesse_geral", "livros_tecnicos", "livros_importados"],
    "instrumento musical": ["instrumentos_musicais"], "audio": ["audio"],
    "alimento": ["alimentos", "alimentos_bebidas"], "bebida": ["bebidas", "alimentos_bebidas"],
    "climatizacao": ["climatizacao"], "ar condicionado": ["climatizacao"],
    "casa e construcao": ["casa_construcao"], "construcao": ["casa_construcao",
                                                            "construcao_ferramentas_construcao"],
    "cool stuff": ["cool_stuff"], "tablet": ["tablets_impressao_imagem"],
    "moveis e artigos de decoracao": ["moveis_decoracao"],
    "roupas infantojuvenis": ["fashion_roupa_infanto_juvenil"],
    "roupa infantojuvenil": ["fashion_roupa_infanto_juvenil"],
    "impressora": ["tablets_impressao_imagem"], "colchao": ["moveis_colchao_e_estofado"],
}
for _cat, _label in CATEGORY_LABELS.items():
    _CATEGORY_TERMS.setdefault(normalize(_label), [_cat])

NEGATIVE = (r"\b(nota baixa|notas baixas|nota 1|nota 2|insatisf\w*|negativ\w*|pessim\w*|pior\w*|"
            r"ruins?)\b")
COMPLAINT = r"\b(reclama\w*|problema\w*|queixa\w*|critica\w*)\b"
# "gostei" e "melhor" ficam de fora: "nao gostei" e "poderia ser melhor" sao reclamacoes
POSITIVE = (r"\b(elogi\w*|valoriz\w*|positiv\w*|satisfeit\w*|nota alta|notas altas|"
            r"nota 5|nota maxima)\b")
LATE = r"\b(atras\w*|demor\w*|fora do prazo|depois do prazo|apos o prazo)\b"
NOT_RECEIVED = (r"\b(nao (recebi|recebeu|receberam|recebido|recebe|chega|chegou|chegaram|entregue|"
                r"entregues|foi entregue|foram entregues)|nunca chegou|extravi\w*|"
                r"antes de receber)\b")
ON_TIME = r"\b(entregues?|entrega|chegou|chegaram) (no|dentro do) prazo\b"


def rule_based_filters(question: str) -> Filters:
    """Filtros por expressoes regulares; usado quando o LLM nao responde ou falha."""
    q = normalize(question)
    f: dict = {}
    if re.search(r"\bnota 3\b", q):
        f["score_min"] = f["score_max"] = 3
    elif re.search(NEGATIVE, q):
        f["score_max"] = 2
    elif re.search(COMPLAINT, q):
        f["score_max"] = 3
    if re.search(POSITIVE, q) and not re.search(r"\binsatisf", q) and "score_min" not in f:
        f.pop("score_max", None)
        f["score_min"] = 4
    if re.search(NOT_RECEIVED, q):
        f["not_received"] = True
    elif re.search(LATE, q):
        f["late"] = True
    elif re.search(ON_TIME, q):
        f["late"] = False

    # termos mais longos primeiro; o trecho encontrado sai da frase, para que
    # "informatica e acessorios" nao acione tambem "informatica"
    cats: list[str] = []
    rest = q
    for term in sorted(_CATEGORY_TERMS, key=len, reverse=True):
        pattern = rf"\b{re.escape(term)}s?\b"
        if re.search(pattern, rest):
            cats.extend(c for c in _CATEGORY_TERMS[term] if c not in cats)
            rest = re.sub(pattern, " ", rest)
    if cats:
        f["categories"] = cats

    years = sorted(set(re.findall(r"\b(20[0-9]{2})\b", q)))
    month = next((m for name, m in _MONTHS.items() if re.search(rf"\b{name}\b", q)), None)
    if years:
        if month and len(years) == 1:
            f["period_start"] = f["period_end"] = f"{years[0]}-{month:02d}"
        else:
            f["period_start"], f["period_end"] = f"{years[0]}-01", f"{years[-1]}-12"
    if re.search(r"\b(black friday)\b", q):
        f["period_start"], f["period_end"] = "2017-11", "2017-11"

    states, rest = [], q
    for name, uf in sorted(_STATE_NAMES.items(), key=lambda x: -len(x[0])):
        if re.search(rf"\b{name}\b", rest):
            states.append(uf)
            rest = re.sub(rf"\b{name}\b", " ", rest)
    states += [uf for uf in STATES if re.search(rf"\b{uf.lower()}\b", q) and uf not in states
               and uf.lower() not in {"es", "pa", "se", "to", "ma", "al", "am", "ap", "pe", "pi",
                                      "ce", "ac", "mg", "ms"}]
    # norte e sul so com "regiao": aparecem em "rio grande do norte" e "mato grosso do sul"
    region = [r for name, r in _REGIONS.items()
              if re.search(rf"\bregiao {name}\b", q)
              or (name in ("nordeste", "sudeste", "centro oeste") and re.search(rf"\b{name}\b", q))]
    if region:
        f["regions"] = region
    elif states:
        f["states"] = sorted(set(states))
    return Filters(**f)
