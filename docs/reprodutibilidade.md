# Reprodutibilidade

## Ambiente

- Python 3.12, dependências fixadas em `uv.lock` (`uv sync`) e exportadas em `requirements.txt`. O torch vem do índice CPU do PyTorch; nada exige GPU.
- Testado num notebook com i7 e 16 GB de RAM, sem GPU.
- Modelos baixados do Hugging Face na primeira execução: `paraphrase-multilingual-MiniLM-L12-v2` (embeddings) e `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (re-ranking). Stopwords e stemmer RSLP do NLTK são baixados pelo próprio código.

## Variáveis de ambiente

Listadas em `.env.example`. Só as etapas que chamam o LLM precisam delas.

| Variável | Uso |
|---|---|
| `GOOGLE_API_KEY` | chave da API do Gemini |
| `GEMINI_CHAT_MODEL` | análise da pergunta, geração e HyDE (`gemini-3.6-flash` nos resultados) |
| `GEMINI_JUDGE_MODEL` | juiz de relevância e de fidelidade (`gemini-3.5-flash`) |
| `GEMINI_FALLBACK_MODEL` | modelo de reserva, opcional |
| `TC_ENV_FILE` | arquivo com as variáveis fora do repositório, opcional |
| `TC_LLM_INTERVALO`, `TC_LLM_PARALELO` | ritmo e chamadas simultâneas ao LLM |
| `TC_RAPIDO` | 1 avalia uma pergunta de cada cinco, para iterar |

## Etapas e o que é versionado

| Etapa | Comando | Tempo em CPU | Versionado |
|---|---|---|---|
| Download | `make data` | 1 min | `data/manifest.json` (os CSVs não) |
| Base | `make base` | 30 s | `data/processed/avaliacoes.parquet` |
| Documentos | `make documentos` | 15 s | `data/processed/documentos.parquet` |
| Índices e temas | `make index` | 10 min | `data/index/` |
| Avaliação | `make avaliacao` | 30 a 60 min, a maior parte em chamadas ao LLM | `data/eval/`, `reports/rastros/avaliacao.parquet` |
| Notebooks | `make notebooks` | 15 min | `notebooks/`, `images/`, `reports/` |

Com os artefatos versionados, a API e os notebooks rodam sem refazer as etapas anteriores. Os notebooks 05 e 06 leem os julgamentos versionados e não chamam o LLM; o notebook 07 chama o LLM (ou o cache) para as perguntas de demonstração.

## Determinismo

- Base, documentos, BM25 e embeddings são determinísticos.
- O mapa de temas usa UMAP com semente 42 e HDBSCAN; na mesma máquina, as contagens se repetiram exatamente entre execuções. Em outra máquina, a numeração dos tópicos pode mudar: a correspondência de tópicos para temas tem uma palavra de controle por tópico e, se ela não bater, o tópico vira "Outros" com aviso, em vez de receber o nome errado.
- A divisão do conjunto de avaliação em calibração e teste usa semente 42.
- As chamadas ao LLM usam temperatura 0, mas a saída de um LLM por API não é garantidamente idêntica entre execuções. Por isso cada resposta fica em cache em `.cache/llm/` (fora do Git), e os resultados que dependem do LLM ficam versionados.

## Custo da avaliação

Uma rodada completa (`make avaliacao`) fez cerca de 430 chamadas ao LLM: 80 para o HyDE, 85 para o julgamento do pool, 30 do segundo juiz, 80 de análise das perguntas, 53 de geração, 53 do juiz de fidelidade, 66 para julgar os textos do pipeline final e 3 do teste de injeção, além das repetições na correção do julgamento de fidelidade. A cota gratuita do Gemini para os modelos usados era de 20 requisições por dia por modelo, o que não comporta a avaliação; ela foi feita com faturamento pré-pago.
