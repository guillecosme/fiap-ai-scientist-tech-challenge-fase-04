# Fontes de dados

## Brazilian E-Commerce Public Dataset by Olist

| Item | Valor |
|---|---|
| Publicação | Kaggle, conjunto `olistbr/brazilian-ecommerce` ([link](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)) |
| Responsável | Olist, com os dados anonimizados pela própria empresa |
| Licença | Creative Commons BY-NC-SA 4.0: uso não comercial, com atribuição e compartilhamento pela mesma licença |
| Versão usada | 2 (registrada em `data/manifest.json`) |
| Período | pedidos de 2016 a 2018, segundo a descrição da base |
| Formato | nove CSVs em UTF-8 |

Os CSVs não são versionados neste repositório. `make data` baixa a base pelo `kagglehub` e grava o SHA-256 e o tamanho de cada arquivo em `data/manifest.json`. Quem reproduzir o projeto confere a própria cópia contra esse manifesto; um hash diferente significa que a versão do Kaggle mudou e os números dos notebooks podem não bater.

### Autenticação no Kaggle

O download exige uma conta no Kaggle. O `kagglehub` procura o token em `~/.kaggle/` ou nas variáveis de ambiente; o código do projeto não lê nem grava credencial. Alternativa sem conta configurada: baixar o zip pela página do conjunto e extrair os CSVs em `data/raw/olist/`. O manifesto continua servindo para conferir os arquivos.

## Tabelas

| Tabela | Arquivo | Grão | Uso no projeto |
|---|---|---|---|
| Avaliações | `olist_order_reviews_dataset.csv` | uma avaliação de um pedido | **base do RAG**: nota, título, comentário, data de envio da pesquisa e data da resposta |
| Pedidos | `olist_orders_dataset.csv` | um pedido | metadados de entrega: status, compra, aprovação, envio à transportadora, entrega e data estimada |
| Itens do pedido | `olist_order_items_dataset.csv` | um item de um pedido | produto, vendedor, preço e frete; liga o pedido à categoria |
| Produtos | `olist_products_dataset.csv` | um produto | categoria em português |
| Tradução de categorias | `product_category_name_translation.csv` | uma categoria | nome da categoria em inglês, usado só para conferência |
| Clientes | `olist_customers_dataset.csv` | um cliente por pedido | UF do cliente |
| Pagamentos | `olist_order_payments_dataset.csv` | um pagamento de um pedido | **fora**: forma de pagamento não aparece entre as perguntas de negócio do desafio, e o valor do pedido já vem dos itens |
| Vendedores | `olist_sellers_dataset.csv` | um vendedor | **fora**: o identificador do vendedor já está nos itens; a localização do vendedor não entra em nenhuma pergunta prevista |
| Geolocalização | `olist_geolocation_dataset.csv` | um CEP com coordenadas | **fora**: a UF do cliente basta para o recorte regional; coordenadas não acrescentam nada à recuperação de texto |

## Chaves de junção

- `order_reviews.order_id` liga a avaliação ao pedido.
- `orders.customer_id` liga o pedido ao cliente. `customer_id` é gerado por pedido; o cliente recorrente é identificado por `customer_unique_id`.
- `order_items.order_id` liga o pedido aos itens; `order_items.product_id` liga o item ao produto e à categoria.

## Cuidados conhecidos da base

Cada um é medido no notebook 01 e tratado em `src/data/base.py`:

- `review_id` repetido em mais de um pedido;
- pedido com mais de uma avaliação;
- pedido com vários itens de categorias diferentes;
- produtos sem categoria;
- comentários com quebras de linha, emojis e texto composto só de pontuação;
- avaliações respondidas antes da entrega do pedido.

## Anonimização dos textos

A Olist anonimizou os comentários trocando os nomes de empresas e parceiros por nomes de casas da série Game of Thrones. Nos títulos e comentários aparecem "Lannister" (1.223 avaliações), "Stark" (305), "Targaryen" (276) e "Baratheon" (275), em frases como "parabéns lojas lannister" ou "vendido e entregue targaryen"; "olist" não aparece nenhuma vez. Para o projeto isso tem duas consequências:

- esses nomes indicam a loja ou o parceiro citado pelo cliente, não um produto, e a resposta ao gestor não pode tratá-los como marcas reais;
- perguntas sobre uma empresa pelo nome real ("o que falam da Olist") ficam fora do que a base consegue sustentar.

"Correios" não foi anonimizado (724 avaliações) e continua útil para as perguntas de entrega.
