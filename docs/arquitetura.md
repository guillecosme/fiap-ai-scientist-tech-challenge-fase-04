# Arquitetura

## Visão geral

A solução tem duas partes. A **preparação** roda uma vez e grava artefatos versionados (base, documentos, índices, temas). A **consulta** roda a cada pergunta, num grafo LangGraph que lê esses artefatos, chama o LLM no máximo duas vezes e devolve a resposta com as evidências.

| Etapa | Módulo | Entrada | Saída |
|---|---|---|---|
| Download | `src/data/download.py` | Kaggle (`kagglehub`) | `data/raw/olist/*.csv`, `data/manifest.json` |
| Base de avaliações | `src/data/base.py` | 5 tabelas da Olist | `data/processed/avaliacoes.parquet` (98.410 linhas) |
| Documentos | `src/preprocessing/` | base | `data/processed/documentos.parquet` (42.093 linhas, 34.889 textos distintos) |
| Índice lexical | `src/indexing/lexical.py` | documentos | `data/index/bm25.parquet` |
| Índice vetorial | `src/indexing/vetorial.py` | documentos | `data/index/embeddings.npz` (float16) |
| Mapa de temas | `src/indexing/temas.py` | documentos e embeddings | `data/index/temas.parquet`, `reports/temas.csv` |
| Consulta | `src/graph/fluxo.py` | pergunta | resposta, evidências e rastro |
| API | `src/api/main.py` | HTTP | JSON |

## Grafo da consulta

```
START ─► analisar ─┬─ escopo "fora" ──────────────────────────────► recusar ─► END
                   └─► recuperar ─► avaliar_evidencia ─┬─ não passa ─► recusar ─► END
                                                       └─► gerar ─┬─ falha do LLM ─────► END
                                                                  └─► verificar ─┬─ insuficiente ─► recusar ─► END
                                                                                 └─► END
```

| Nó | O que faz | Chamadas ao LLM |
|---|---|---|
| `analisar` | saída estruturada `Analise`: escopo, pergunta de frequência, consulta reescrita e `Filters`. Se a chamada falha, `rule_based_filters` preenche os filtros | 1 |
| `recuperar` | máscara de avaliações pelos filtros; BM25 e FAISS (50 cada) entre as entradas permitidas; RRF; cross-encoder; MMR para 10; contagem por tema no recorte quando a pergunta é de frequência | 0 |
| `avaliar_evidencia` | `Gate.decide`: recorte vazio, menos de 3 avaliações no recorte, ou maior score do cross-encoder abaixo de -0,37 | 0 |
| `gerar` | monta as evidências E1 a E10 e chama a cadeia LCEL `PROMPT | LLM com schema Resposta` | 1 |
| `verificar` | `verify`: retira afirmações sem citação válida; resposta sem afirmação válida vira insuficiente | 0 |
| `recusar` | mensagem de recusa pelo motivo: fora do escopo, sem avaliações, poucas avaliações, evidência insuficiente ou insuficiência apontada pelo LLM | 0 |

O estado do grafo (`Estado`) carrega a pergunta, a análise, os filtros, o recorte, as entradas recuperadas com os scores, a frequência, a decisão do gate, as evidências, a resposta, a verificação, o tempo de cada etapa e os erros. `src/graph/rastro.py` transforma o estado final numa linha de Parquet.

## Recuperação

- **Filtros** (`src/retrieval/filtros.py`): aplicados por avaliação. Uma entrada do índice (texto distinto) concorre quando pelo menos uma das suas avaliações passa; a contagem exibida é a das que passaram.
- **BM25** (`src/indexing/lexical.py`): `rank_bm25` com k1 = 1,5 e b = 0,75 sobre radicais RSLP, mantendo as negações.
- **Vetorial** (`src/indexing/vetorial.py`): FAISS `IndexFlatIP` com vetores normalizados; o filtro entra como `IDSelectorBatch`.
- **Fusão** (`src/retrieval/hibrida.py`): Reciprocal Rank Fusion com k = 60.
- **Re-ranking** (`src/retrieval/reranking.py`): cross-encoder `mmarco-mMiniLMv2-L12-H384-v1` sobre os 50 candidatos.
- **Diversidade** (`src/retrieval/mmr.py`): MMR com λ = 0,7.

## Geração

- **Provedor** (`src/generation/provedor.py`): Gemini via `langchain-google-genai`, modelo por variável de ambiente, modelo de reserva opcional, cache em disco por hash do prompt, retentativa com espera crescente em 429 e 5xx, intervalo mínimo entre chamadas.
- **Prompt** (`src/generation/prompts.py`): regras de fundamentação, evidências delimitadas como dado, tabela de frequência com o denominador.
- **Schemas** (`src/generation/esquemas.py`): `Analise`, `Resposta` e `Afirmacao`.
- **Citações** (`src/generation/citacoes.py`): verificação por código.

## API

`make api` sobe o FastAPI em `http://127.0.0.1:8000`, com documentação interativa em `/docs`.

| Método | Rota | Corpo | Resposta |
|---|---|---|---|
| GET | `/saude` | | `status`, `pipeline_carregado` |
| POST | `/perguntar` | `{"pergunta": "..."}` (3 a 500 caracteres) | `decisao`, `motivo`, `resumo`, `afirmacoes` (texto e ids das evidências), `limitacoes`, `recorte`, `avaliacoes_no_recorte`, `frequencia_por_tema`, `evidencias` (só as citadas: texto, nota, data, categoria, entrega, avaliações com o mesmo texto, `review_ids`), `tempos_s`, `chamadas_llm` |

O pipeline é carregado na subida, com uma busca de aquecimento. Cada consulta grava uma linha em `reports/rastros/api.parquet`, fora do Git.

## Interface

`make app` abre a interface Streamlit (`app.py`) em `http://localhost:8501`. Ela carrega o mesmo `Pipeline` da API uma vez (`st.cache_resource`), com a busca de aquecimento, e tem três abas: perguntar (resposta, recorte, frequência por tema, afirmações, evidências citadas e rastro da consulta), qualidade medida (métricas lidas de `reports/` e `data/eval/`) e auditoria do juiz (rotulação dos pares de `data/eval/auditoria_manual.csv`, com a concordância calculada na hora). A lógica de apresentação fica em `src/app/servicos.py` e é testada em `tests/test_app.py`, inclusive com a tela executada pelo `AppTest` do Streamlit sobre um pipeline falso.
