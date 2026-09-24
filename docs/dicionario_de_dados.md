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
