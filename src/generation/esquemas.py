"""Schemas de saida estruturada do LLM (Pydantic)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.retrieval.filtros import Filters


class Afirmacao(BaseModel):
    """Uma afirmacao da resposta e as evidencias que a sustentam (D11)."""

    texto: str = Field(description="uma afirmação curta, em português, para o gestor")
    evidencias: list[str] = Field(
        description="ids das evidências que sustentam a afirmação, como E1 ou E4; "
        "pelo menos um")


class Resposta(BaseModel):
    suficiente: bool = Field(
        description="false quando as evidências não bastam para responder à pergunta")
    resumo: str = Field(description="resposta direta em uma ou duas frases")
    afirmacoes: list[Afirmacao] = Field(
        description="os achados, cada um com as evidências; vazio se suficiente for false")
    limitacoes: str = Field(
        description="o que as evidências não cobrem ou o cuidado na leitura; pode ser vazio")


class Analise(BaseModel):
    """Leitura da pergunta antes da busca."""

    escopo: Literal["dentro", "fora"] = Field(
        description="dentro: a pergunta é sobre a experiência de compra relatada nas "
        "avaliações de clientes (entrega, produto, atendimento, loja, elogios, problemas). "
        "fora: pede dado que avaliações não contêm (faturamento, lucro, funcionários, "
        "métricas do site) ou assunto alheio a compras")
    frequencia: bool = Field(
        description="true se a pergunta pede os principais, os mais frequentes ou um ranking")
    consulta: str = Field(
        description="a pergunta reescrita como um cliente escreveria numa avaliação, para a "
        "busca, sem as restrições de nota, categoria, período ou região")
    filtros: Filters = Field(description="recorte da base pedido pela pergunta")
