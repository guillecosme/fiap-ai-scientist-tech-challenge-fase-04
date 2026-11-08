# Métricas do pipeline

Gerado pelo notebook 06 a partir de `reports/rastros/avaliacao.parquet` e dos julgamentos em `data/eval/`. Conjunto de avaliação: 80 perguntas (60 respondíveis, 20 sem resposta possível), metade de calibração e metade de teste.

## Recuperação (60 perguntas respondíveis, 10 primeiros resultados)

| Configuração | recall@10 | precision@10 | MRR |
|---|---|---|---|
| bm25 | 0.199 | 0.328 | 0.490 |
| vetorial | 0.240 | 0.385 | 0.554 |
| hibrida | 0.230 | 0.378 | 0.561 |
| hibrida_rerank | 0.306 | 0.472 | 0.648 |
| hibrida_rerank_filtros | 0.500 | 0.693 | 0.860 |
| hibrida_rerank_filtros_hyde | 0.509 | 0.688 | 0.815 |
| pipeline final | 0.420 | 0.882 | 0.956 |

O pipeline final usa os filtros extraídos pelo LLM, a consulta reescrita e MMR. Foi medido com os julgamentos do pool mais 534 pares que só ele trouxe; precisão e MRR são comparáveis às variantes, mas o recall usa um conjunto de relevantes maior e não é.

## Geração

- respostas avaliadas: 53; afirmações: 169
- fidelidade (afirmações sustentadas pelas evidências citadas e pela tabela de frequência, juiz LLM): 0.959
- fidelidade com o juiz sem acesso à tabela de frequência: 0.728 (as afirmações de frequência não podiam ser conferidas)
- respostas com todas as afirmações sustentadas: 0.906
- relevância média da resposta (0 a 2): 2.00
- validade das citações (código): 0.970

## Recusa (perguntas de teste)

- respondíveis respondidas: 27 de 29
- não respondíveis recusadas: 10 de 10

## Operação (perguntas respondidas)

- tempo mediano estimado por resposta: 5.6 s (análise 1.6 s, recuperação 1.0 s, geração 3.0 s)
- chamadas ao LLM por resposta: 2.0; tokens de entrada 1413 e de saída 449 (medianas)
