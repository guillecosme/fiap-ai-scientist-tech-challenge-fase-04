# Voice of Customer Intelligence com RAG

Projeto da Fase 4 do MBA em AI Scientist (FIAP / POSTECH). A proposta é permitir que gestores de e-commerce façam perguntas em linguagem natural sobre as avaliações de clientes da Olist e recebam respostas sustentadas pelos comentários recuperados da base, com as avaliações usadas como evidência visíveis na resposta e recusa explícita quando a base não traz evidência suficiente.

## Contexto do problema

Uma operação de e-commerce acumula milhares de avaliações. A nota de 1 a 5 mostra o nível de satisfação, mas o motivo da nota está no comentário: atraso, produto diferente do anunciado, defeito, pedido não recebido, atendimento do vendedor, elogios à rapidez ou à qualidade. Ler esses comentários um a um não escala, e perguntar diretamente a um modelo de linguagem, sem acesso à base, produz respostas genéricas ou sem fundamento.

Retrieval-Augmented Generation (RAG) trata os dois pontos. Primeiro recupera as avaliações relevantes para a pergunta; depois gera a resposta usando apenas esse conteúdo como contexto. Com isso, cada afirmação da resposta pode ser conferida contra as avaliações que a sustentam.

## Perguntas que a solução deve responder

Exemplos do enunciado:

- Quais são os principais problemas relatados pelos clientes?
- Quais reclamações relacionadas à entrega aparecem com maior frequência?
- O que os clientes mais elogiam em suas experiências de compra?
- Quais problemas são relatados por clientes insatisfeitos?
- Existem padrões recorrentes nas avaliações relacionadas à qualidade dos produtos?

Com os metadados do pedido (desafio adicional), a busca também passa a responder perguntas restritas a um recorte, como os problemas relatados nas avaliações com nota baixa, os elogios em uma categoria de produto ou os temas das avaliações de entregas atrasadas.

Quando não houver avaliações relevantes, quando a evidência recuperada for insuficiente ou quando a pergunta estiver fora do escopo da base, a solução deve informar que não há evidência suficiente, sem gerar uma resposta que os dados não sustentam.

## Objetivo

Construir um pipeline RAG completo, dos CSVs da Olist à resposta com evidências, que:

1. organize as avaliações em uma base de conhecimento, com limpeza do texto, tratamento de valores ausentes, metadados do pedido, embeddings e índices, cada escolha justificada;
2. recupere as avaliações mais relevantes combinando busca lexical (BM25) e semântica (embeddings), com filtros por nota, período, categoria e situação da entrega;
3. gere respostas em que cada afirmação cita as avaliações que a sustentam;
4. reconheça as perguntas sem evidência suficiente e recuse de forma explícita;
5. meça tudo isso com um conjunto de avaliação definido antes da construção: qualidade da recuperação, fidelidade das respostas às evidências, validade das citações e acerto da recusa.

## Base de dados

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), publicada no Kaggle pela Olist sob licença CC BY-NC-SA 4.0. Segundo a descrição da base, são cerca de 100 mil pedidos feitos entre 2016 e 2018. O projeto usa as avaliações (nota, título, comentário e datas) e, como metadados, as tabelas de pedidos, itens, produtos, categorias e clientes.

Os CSVs não são redistribuídos neste repositório. `make data` baixa a base do Kaggle e grava o SHA-256 de cada arquivo em `data/manifest.json`, para que qualquer cópia local possa ser conferida.

## Etapas do projeto

O trabalho segue a ordem abaixo. Cada etapa tem um notebook, e o código reutilizável fica em `src/`.

| Etapa | Notebook | O que entrega |
|---|---|---|
| Base e qualidade | `notebooks/01_base_e_qualidade.ipynb` | base de avaliações com os metadados do pedido, checagens de qualidade e hipóteses sobre quem comenta, nota e atraso |
| EDA textual | `notebooks/02_eda_textual.ipynb` | tamanho dos comentários, termos discriminativos por nota e decisões de preparação do texto |
| Representação e índices | `notebooks/03_representacao_e_indices.ipynb` | índice lexical BM25, embeddings multilíngues e índice vetorial FAISS |
| Mapa de temas | `notebooks/04_mapa_de_temas.ipynb` | temas das avaliações com BERTopic, comparação com LDA e frequência dos temas por nota |
| Avaliação da recuperação | `notebooks/05_avaliacao_recuperacao.ipynb` | comparação das estratégias de busca com recall@k, precision@k e MRR |
| Geração, recusa e fundamentação | `notebooks/06_geracao_recusa_e_fundamentacao.ipynb` | fidelidade das respostas, validade das citações e acerto da recusa |
| Demonstração | `notebooks/07_demonstracao.ipynb` | perguntas do enunciado respondidas com as evidências recuperadas |

## Fluxo previsto

Preparação, feita uma vez (`make data`, `make base`, `make documentos`, `make index`):

1. leitura dos CSVs e montagem da base com uma linha por avaliação e os metadados do pedido;
2. limpeza e normalização dos comentários, preservando o texto original para exibição;
3. construção do índice lexical (BM25), do índice vetorial (FAISS) e do mapa de temas.

Consulta, a cada pergunta:

1. análise da pergunta: escopo, filtros por metadado e reescrita para a busca;
2. recuperação híbrida: BM25 e busca vetorial combinadas por Reciprocal Rank Fusion, seguidas de re-ranking;
3. avaliação da evidência recuperada, com limiar calibrado no conjunto de avaliação;
4. geração da resposta com citação das avaliações, ou recusa explícita quando a evidência não basta.

O fluxo da consulta é orquestrado com LangGraph e exposto por uma API FastAPI. A geração usa o Gemini por meio do LangChain.

## Estrutura do repositório

```
.
├── data/               raw (baixado do Kaggle, fora do Git), processed, index e eval
├── notebooks/          um notebook por etapa, numerados de 01 a 07
├── src/
│   ├── data/           download com manifesto, leitura e junção das tabelas
│   ├── preprocessing/  limpeza, normalização e deduplicação dos comentários
│   ├── indexing/       BM25, embeddings, FAISS e mapa de temas
│   ├── retrieval/      filtros por metadado, busca híbrida, fusão e re-ranking
│   ├── generation/     prompts, saída estruturada com citações e provedor do modelo
│   ├── graph/          fluxo da consulta em LangGraph, com a avaliação da evidência
│   ├── evaluation/     métricas de recuperação, geração e recusa
│   └── api/            API FastAPI
├── tests/              testes de junções, limpeza, filtros, fusão, citações e API
├── reports/            métricas, ablação, exemplos de perguntas e respostas, slides
├── images/             figuras exportadas dos notebooks
├── docs/               arquitetura, metodologia, decisões, dicionário de dados e fontes
├── Makefile            atalhos: setup, data, base, documentos, index, notebooks, test, lint
├── pyproject.toml      dependências gerenciadas pelo uv
├── requirements.txt    exportado do lock, para quem preferir pip
└── .env.example        variáveis de ambiente esperadas
```

## Como rodar

O projeto roda em CPU. O `pyproject.toml` aponta o torch para o índice CPU do PyTorch, o que evita baixar as bibliotecas CUDA.

```bash
uv sync                 # ou: pip install -r requirements.txt
cp .env.example .env    # preencher a chave do Gemini e os nomes dos modelos
make data               # baixa a base da Olist (exige credencial do Kaggle)
make base               # monta a base de avaliações
make documentos         # prepara os documentos do índice
make index              # constrói os índices e o mapa de temas
make notebooks          # executa os notebooks na ordem
make test               # roda os testes
```

As seções de preparação da base, estratégia de recuperação, avaliação, exemplos de perguntas e respostas, decisões e limitações são preenchidas conforme as etapas avançam.
