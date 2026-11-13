# Metodologia

O projeto segue a ordem dos notebooks: entender a base (01), o texto (02), representar e indexar (03), mapear temas (04), medir a recuperação (05), medir a geração e a recusa (06) e demonstrar (07). Cada decisão está em `docs/decisoes_analiticas.md` com a evidência que a sustenta.

## Base de conhecimento

1. **Grão.** A avaliação (`review_id`). Uma avaliação ligada a vários pedidos do mesmo cliente aparece uma vez, com os pedidos agregados; pedidos com várias avaliações mantêm todas.
2. **Metadados.** Situação da entrega, dias de atraso contra a data estimada, resposta antes da entrega, categoria principal e lista de categorias, período da compra, UF e região, construídos em `src/data/base.py` e testados em `tests/test_base.py`.
3. **Hipóteses com teste estatístico** (notebooks 01 e 02), com o tamanho do efeito como critério: quem comenta dá nota diferente (V de Cramér 0,29); atraso derruba a nota (correlação bisserial de postos 0,64); quem responde antes de receber reclama da entrega (0,76); comentários negativos são mais longos (0,51); vocabulário de entrega aparece em elogios; o mesmo problema é escrito de formas diferentes; os temas diferem entre categorias (V de Cramér 0,16).
4. **Texto.** Três versões (original, limpo, normalizado), um documento por avaliação sem fragmentação, deduplicação de textos idênticos com contagem.
5. **Índices.** BM25 sobre radicais RSLP com as negações mantidas; embeddings `paraphrase-multilingual-MiniLM-L12-v2` em FAISS `IndexFlatIP`; mapa de temas com BERTopic, um modelo por polaridade, comparado com LDA pela coerência c_v.

## Recuperação

Filtros por metadado antes da busca, BM25 e vetorial com 50 candidatos cada, fusão por Reciprocal Rank Fusion (k = 60), re-ranking por cross-encoder e MMR (λ = 0,7) nos 10 que vão ao LLM. Cada componente entrou ou saiu pela ablação do notebook 05; o HyDE saiu por não trazer ganho.

## Geração e recusa

A pergunta passa por uma análise com saída estruturada (escopo, frequência, consulta reescrita e filtros). Perguntas fora do escopo são recusadas antes da busca. Depois da busca, o gate recusa quando o recorte é vazio ou pequeno ou quando o maior score do cross-encoder fica abaixo do limiar calibrado. A geração recebe as evidências numeradas, delimitadas e marcadas como dado, e a tabela de frequência por tema contada em código; cada afirmação cita as evidências, e a citação é conferida por código.

## Avaliação

A avaliação foi definida antes da construção do pipeline e não ganhou perguntas ou métricas ao longo das rodadas. O conjunto de perguntas é a única parte construída do projeto; todo o resto é dado real da Olist.

### Conjunto de perguntas

`data/eval/perguntas.csv` tem 80 perguntas:

| Tipo | Quantidade | O que se espera do sistema |
|---|---|---|
| `respondivel` | 60 | resposta com evidências |
| `fora_do_escopo` | 8 | recusa: a pergunta pede algo que avaliações de clientes não contêm (faturamento, clima, receitas) |
| `sem_evidencia` | 8 | recusa: o assunto é de e-commerce, mas não aparece na base (PIX, criptomoeda, cashback, compras de 2020). A ausência foi conferida por busca no texto normalizado: zero menções |
| `evidencia_insuficiente` | 4 | recusa ou aviso de evidência insuficiente: categorias com uma ou duas avaliações com texto, ou assunto com duas menções |

Entre as respondíveis, 15 vêm do enunciado do desafio, 25 tratam de temas específicos (produto errado, defeito, embalagem, nota fiscal, atendimento) e 20 dependem de metadado (nota, categoria, período, região, atraso). Cada pergunta traz:

- `filtros_esperados`: o recorte explícito pedido pela pergunta (categoria, período, UF, região, atraso, não recebido, nota quando a pergunta fala em nota ou em satisfação). É usado para conferir o recorte das evidências e a extração de filtros;
- `alvo_objetivo`: para as perguntas sobre atraso e não recebimento, o grupo de metadado que permite medir a precisão da recuperação sem juiz;
- `split`: metade das perguntas de cada tipo e origem é de calibração (usada para escolher o limiar do gate de evidência) e metade é de teste (onde a recusa é medida). A divisão é aleatória com semente fixa.

### Relevância

A relevância é julgada por pooling: para cada pergunta, os 10 primeiros resultados de cada variante de busca (BM25, vetorial, híbrida, híbrida com re-ranking, com filtros e com HyDE) formam o pool, e só o pool é julgado. O que nenhuma variante trouxe conta como não relevante para todas, o que mantém a comparação justa entre elas. O recall é medido contra os relevantes conhecidos do pool.

A relevância tem duas partes:

1. **Assunto, julgado pelo LLM** (`src/evaluation/julgamento.py`), numa rubrica de três níveis: 2 (relevante, serviria de evidência), 1 (parcial, toca no assunto de forma vaga) e 0 (irrelevante). O juiz não vê a nota nem os metadados da avaliação e é instruído a ignorar as restrições de categoria, período e região da pergunta, que o texto não revela. As avaliações entram delimitadas e marcadas como dado, não instrução. O juiz é um modelo diferente do gerador.
2. **Recorte, conferido por código**: a entrada só conta como relevante se ao menos uma das suas avaliações atende aos `filtros_esperados` da pergunta.

Relevante, para as métricas principais, é nota 2 dentro do recorte; a leitura tolerante (nota 1 ou 2) aparece como análise de sensibilidade.

### Auditoria do juiz

Uma amostra de 120 pares (30 perguntas respondíveis sorteadas, 4 pares de cada, priorizando notas diferentes do juiz) fica em `data/eval/auditoria_manual.csv`. A amostra recebe:

- a nota de um segundo juiz, de outra família de modelos (`gemma-4-26b-a4b-it`), para medir a concordância entre juízes automáticos (kappa de Cohen);
- uma coluna `nota_manual` para a anotação humana do autor. Quando preenchida, o notebook 05 calcula a concordância do juiz com ela.

### Métricas

| Bloco | Métricas |
|---|---|
| Recuperação | recall@10, precision@10 e MRR por variante, com intervalo de confiança de 95% por bootstrap pareado das perguntas; precisão objetiva por metadado nas perguntas de atraso e de não recebimento |
| Geração | fidelidade (afirmações sustentadas pelas evidências citadas, julgadas pelo LLM), relevância da resposta e validade das citações (conferida por código) |
| Recusa | matriz de confusão do gate nas perguntas de teste: recusa correta nas não respondíveis, recusa indevida nas respondíveis |
| Operação | latência por etapa, chamadas e tokens do LLM por consulta, a partir do rastro |

### Cuidados com vazamento

- As regras de extração de filtros (`rule_based_filters`) foram ajustadas lendo as 80 perguntas do conjunto; a concordância delas com `filtros_esperados` é otimista e não é usada como resultado. O extrator principal, por LLM, não foi ajustado nessas perguntas.
- O limiar do gate é escolhido só nas perguntas de calibração e medido só nas de teste.
- O juiz não conhece a variante que trouxe cada avaliação.
