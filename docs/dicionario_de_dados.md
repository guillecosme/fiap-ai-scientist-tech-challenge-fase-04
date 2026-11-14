# Dicionário de dados

## `data/processed/avaliacoes.parquet`

Base de avaliações gerada por `make base` (`src/data/base.py`). Uma linha por avaliação (`review_id`), com os metadados dos pedidos que ela cobre. As regras de junção e agregação estão no cabeçalho do módulo e são testadas em `tests/test_base.py`.

### Avaliação

| Coluna | Tipo | Descrição |
|---|---|---|
| `review_id` | texto | identificador da avaliação; chave única da base |
| `review_score` | inteiro | nota de 1 a 5 dada pelo cliente |
| `review_comment_title` | texto | título do comentário, como escrito pelo cliente; ausente quando não preenchido |
| `review_comment_message` | texto | comentário, como escrito pelo cliente (com quebras de linha e emojis originais); ausente quando não preenchido |
| `review_creation_date` | data | dia em que a pesquisa de satisfação foi enviada ao cliente |
| `review_answer_timestamp` | data e hora | momento em que o cliente respondeu |
| `answer_days` | decimal | dias entre o envio da pesquisa e a resposta |
| `has_comment` | booleano | comentário preenchido com algum caractere além de espaço |
| `has_title` | booleano | título preenchido com algum caractere além de espaço |

### Pedido e cliente

| Coluna | Tipo | Descrição |
|---|---|---|
| `order_ids` | lista de texto | pedidos cobertos pela avaliação (quase sempre um) |
| `n_orders` | inteiro | quantidade de pedidos em `order_ids` |
| `order_has_multiple_reviews` | booleano | algum pedido da avaliação recebeu mais de uma avaliação |
| `customer_unique_id` | texto | cliente; o mesmo em todos os pedidos de uma avaliação |
| `customer_state` | texto | UF do cliente |
| `customer_region` | texto | região do Brasil derivada da UF |
| `order_status` | texto | status do pedido na Olist (`delivered`, `shipped`, `canceled`, `unavailable`, `invoiced`, `processing`, `created`, `approved`). Com vários pedidos, o primeiro status diferente de `delivered`, se houver |
| `order_purchase_timestamp` | data e hora | compra; com vários pedidos, a mais antiga |
| `order_approved_at` | data e hora | aprovação do pagamento |
| `order_delivered_carrier_date` | data e hora | entrega do pacote à transportadora |
| `order_delivered_customer_date` | data e hora | entrega ao cliente; ausente se algum pedido da avaliação não foi entregue |
| `order_estimated_delivery_date` | data | data estimada de entrega informada na compra |

### Itens e categoria

| Coluna | Tipo | Descrição |
|---|---|---|
| `n_items` | inteiro | itens somados dos pedidos; zero quando o pedido não tem itens na base |
| `n_sellers` | inteiro | vendedores distintos entre os itens |
| `items_value` | decimal | soma dos preços dos itens, em reais |
| `freight_value` | decimal | soma dos fretes, em reais |
| `main_category` | texto | categoria com o maior valor somado entre os itens (empate pela ordem alfabética); ausente quando nenhum item tem categoria |
| `categories` | lista de texto | todas as categorias dos itens, em ordem alfabética |
| `n_categories` | inteiro | quantidade de categorias distintas |

### Entrega e período

| Coluna | Tipo | Descrição |
|---|---|---|
| `delivered` | booleano | há data de entrega ao cliente |
| `delivery_days` | decimal | dias entre a compra e a entrega |
| `estimated_days` | decimal | dias entre a compra e a data estimada |
| `delay_days` | inteiro | dias de calendário entre a data estimada e a entrega; positivo é atraso, zero ou negativo é no prazo; ausente sem entrega |
| `is_late` | booleano | entregue depois do dia estimado; ausente quando não houve entrega |
| `delivery_status` | texto | `no prazo`, `atrasada`, `cancelada` (status `canceled` ou `unavailable`) ou `nao entregue` (demais casos sem entrega) |
| `answered_before_delivery` | booleano | o cliente respondeu antes de receber o pedido, ou o pedido nunca foi entregue |
| `purchase_year` | inteiro | ano da compra |
| `purchase_month` | texto | mês da compra no formato `AAAA-MM` |

## `data/processed/documentos.parquet`

Documentos do índice, gerados por `make documentos` (`src/preprocessing/documentos.py`). Uma linha por avaliação com título ou comentário que tenha pelo menos uma palavra útil depois da limpeza: 42.093 linhas.

| Coluna | Tipo | Descrição |
|---|---|---|
| `review_id` | texto | avaliação |
| `review_comment_title`, `review_comment_message` | texto | título e comentário originais, usados para exibir a evidência |
| `text_clean` | texto | título e comentário juntos, com quebras de linha, emojis e pontuação repetida tratados; entra nos embeddings e no contexto do LLM |
| `text_norm` | texto | minúsculas, sem acento, só letras e dígitos; chave da deduplicação |
| `tokens` | lista de texto | radicais RSLP sem stopwords, com as negações mantidas; entrada do BM25 |
| `n_chars`, `n_words` | inteiro | tamanho do texto limpo |
| `doc_id` | texto | identificador do texto normalizado (12 caracteres do SHA-1); avaliações com o mesmo texto compartilham o `doc_id` |
| `doc_size` | inteiro | avaliações com o mesmo `doc_id` |
| `is_representative` | booleano | avaliação mais antiga do grupo; o texto dela vai para o índice |
| demais colunas | | metadados copiados da base de avaliações para os filtros: nota, datas, pedidos, período, situação da entrega, atraso, resposta antes da entrega, categorias, UF e região |

## `data/index/`

| Arquivo | Conteúdo |
|---|---|
| `bm25.parquet`, `bm25.json` | radicais de cada texto distinto, na ordem do índice, e os parâmetros k1 e b; o BM25 é reconstruído na carga |
| `embeddings.npz`, `embeddings.json` | 34.889 vetores de 384 dimensões em float16 e os `doc_id` na mesma ordem; modelo e normalização no JSON |
| `temas.parquet` | polaridade, tópico do BERTopic e tema nomeado de cada texto distinto |

## `data/eval/`

| Arquivo | Conteúdo |
|---|---|
| `perguntas.csv` | 80 perguntas com tipo, origem, recorte esperado, grupo de metadado e split |
| `hyde.parquet` | avaliação hipotética gerada para cada pergunta (variante HyDE) |
| `pool.parquet` | os 10 primeiros de cada variante de busca por pergunta |
| `julgamentos.parquet`, `julgamentos_pipeline.parquet` | nota do juiz (0 a 2) para cada par pergunta e texto do pool e do pipeline final |
| `auditoria_manual.csv` | amostra de 120 pares com a nota do juiz, a do segundo juiz e a coluna para a anotação humana |
| `gate_sinais.parquet`, `limiar_evidencia.json` | análise e maior score de cada pergunta; limiar calibrado |
| `fidelidade.parquet`, `fidelidade_sem_tabela.parquet` | julgamento de cada afirmação das respostas, com e sem a tabela de frequência no contexto do juiz |
| `teste_injecao.parquet` | resposta de cada pergunta respondida com uma evidência sintética que traz uma instrução, nas duas variantes, e se a resposta citou ou seguiu a instrução |
| `perguntas_adversariais.parquet` | decisão e resposta do fluxo completo para seis perguntas que tentam tirar o sistema das regras |
