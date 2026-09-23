"""Base de avaliacoes: uma linha por avaliacao, com os metadados do pedido.

Uso:
    python -m src.data.base      # le data/raw/olist e grava data/processed/avaliacoes.parquet

Regras de juncao (medidas no notebook 01):
- o grao e o review_id. Um mesmo review_id pode aparecer em mais de um pedido do
  mesmo cliente com nota, texto e datas identicos: e uma unica avaliacao cobrindo
  varios pedidos. Os pedidos ficam em `order_ids` e os metadados sao agregados;
- um pedido com mais de uma avaliacao mantem todas, porque cada uma e uma resposta
  distinta do cliente; a coluna `order_has_multiple_reviews` marca esses casos;
- categoria principal = categoria com o maior valor somado entre os itens dos
  pedidos da avaliacao (empate resolvido pela ordem alfabetica); a lista completa
  fica em `categories`;
- datas de pedidos agregados: compra mais antiga, entrega e estimativa mais tardias.
  Se algum pedido nao foi entregue, a avaliacao conta como nao entregue.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.olist import RAW_DIR, read_table

OUTPUT = Path("data/processed/avaliacoes.parquet")

REVIEW_COLUMNS = [
    "review_id", "review_score", "review_comment_title", "review_comment_message",
    "review_creation_date", "review_answer_timestamp",
]


def _lists(grouped, sort: bool = False) -> pd.Series:
    """Uma lista Python por grupo. O agg(list) do pandas 3 devolve um tipo do Arrow
    que o Parquet grava mas nao consegue ler de volta."""
    data = {key: [str(v) for v in values] for key, values in grouped}
    if sort:
        data = {key: sorted(values) for key, values in data.items()}
    return pd.Series(data, dtype=object)


def load_tables(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    names = ["order_reviews", "orders", "order_items", "products", "customers"]
    return {name: read_table(name, raw_dir) for name in names}


def _order_level(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pedido com cliente: uma linha por order_id."""
    orders = tables["orders"].merge(
        tables["customers"][["customer_id", "customer_unique_id", "customer_state"]],
        on="customer_id", how="left", validate="many_to_one",
    )
    return orders.drop(columns="customer_id")


def _aggregate_orders(pairs: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Agrega os pedidos de cada avaliacao (quase sempre um so)."""
    po = pairs.merge(orders, on="order_id", how="left", validate="many_to_one")
    po = po.sort_values(["review_id", "order_purchase_timestamp", "order_id"])
    g = po.groupby("review_id", sort=False)

    def status(s: pd.Series) -> str:
        values = list(dict.fromkeys(s.dropna()))
        not_delivered = [v for v in values if v != "delivered"]
        return not_delivered[0] if not_delivered else "delivered"

    agg = pd.DataFrame({
        "order_ids": _lists(g["order_id"]),
        "customer_unique_id": g["customer_unique_id"].first(),
        "customer_state": g["customer_state"].first(),
        "order_status": g["order_status"].agg(status),
        "order_purchase_timestamp": g["order_purchase_timestamp"].min(),
        "order_approved_at": g["order_approved_at"].max(),
        "order_delivered_carrier_date": g["order_delivered_carrier_date"].max(),
        # max com skipna=False: se algum pedido nao foi entregue, a entrega fica ausente
        "order_delivered_customer_date": g["order_delivered_customer_date"].agg(
            lambda s: s.max(skipna=False)),
        "order_estimated_delivery_date": g["order_estimated_delivery_date"].max(),
    })
    agg["n_orders"] = agg["order_ids"].apply(len)
    return agg


def _aggregate_items(pairs: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Itens, valor, vendedores e categorias de cada avaliacao."""
    items = tables["order_items"].merge(
        tables["products"][["product_id", "product_category_name"]],
        on="product_id", how="left", validate="many_to_one",
    )
    pi = pairs.merge(items, on="order_id", how="inner")
    g = pi.groupby("review_id")
    agg = pd.DataFrame({
        "n_items": g.size(),
        "n_sellers": g["seller_id"].nunique(),
        "items_value": g["price"].sum(),
        "freight_value": g["freight_value"].sum(),
    })
    by_cat = (
        pi.dropna(subset=["product_category_name"])
        .groupby(["review_id", "product_category_name"], as_index=False)["price"].sum()
        .sort_values(["review_id", "price", "product_category_name"],
                     ascending=[True, False, True])
    )
    agg["main_category"] = by_cat.drop_duplicates("review_id").set_index("review_id")[
        "product_category_name"]
    agg["categories"] = _lists(by_cat.groupby("review_id")["product_category_name"], sort=True)
    return agg


def build_base(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    reviews = tables["order_reviews"]
    pairs = reviews[["review_id", "order_id"]].drop_duplicates()

    # o conteudo da avaliacao e identico entre os pedidos que ela cobre
    content = reviews[REVIEW_COLUMNS].drop_duplicates("review_id").set_index("review_id")
    orders = _aggregate_orders(pairs, _order_level(tables))
    items = _aggregate_items(pairs, tables)

    base = content.join(orders, how="left").join(items, how="left")
    base["n_items"] = base["n_items"].fillna(0).astype(int)
    base["n_sellers"] = base["n_sellers"].fillna(0).astype(int)
    base["categories"] = base["categories"].apply(lambda c: c if isinstance(c, list) else [])
    base["n_categories"] = base["categories"].apply(len)

    multi = reviews.groupby("order_id")["review_id"].nunique()
    multi_orders = set(multi[multi > 1].index)
    base["order_has_multiple_reviews"] = base["order_ids"].apply(
        lambda ids: any(o in multi_orders for o in ids))

    for col, flag in [("review_comment_message", "has_comment"),
                      ("review_comment_title", "has_title")]:
        base[flag] = base[col].fillna("").str.strip().ne("")

    return base.reset_index()


def main() -> None:
    base = build_base(load_tables())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(OUTPUT, index=False, compression="zstd")
    print(f"{len(base):,} avaliacoes, {base['has_comment'].sum():,} com comentario -> {OUTPUT}")


if __name__ == "__main__":
    main()
