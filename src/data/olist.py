"""Leitura dos nove CSVs da Olist com tipos e datas tratados.

Cuidados aplicados em todas as tabelas:
- identificadores e prefixos de CEP como texto (o CEP perde o zero a esquerda se
  for lido como numero);
- apenas o campo vazio conta como ausente. Com os valores padrao do pandas, um
  comentario escrito "NA" ou "null" seria lido como nulo;
- colunas de data convertidas para datetime;
- a tabela de traducao de categorias vem com BOM no inicio, lido com utf-8-sig;
- as colunas de produto com o erro de digitacao "lenght" ganham o nome correto.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/olist")

TABLES: dict[str, dict] = {
    "customers": {
        "file": "olist_customers_dataset.csv",
        "text": ["customer_id", "customer_unique_id", "customer_zip_code_prefix",
                 "customer_city", "customer_state"],
    },
    "geolocation": {
        "file": "olist_geolocation_dataset.csv",
        "text": ["geolocation_zip_code_prefix", "geolocation_city", "geolocation_state"],
    },
    "order_items": {
        "file": "olist_order_items_dataset.csv",
        "text": ["order_id", "product_id", "seller_id"],
        "dates": ["shipping_limit_date"],
    },
    "order_payments": {
        "file": "olist_order_payments_dataset.csv",
        "text": ["order_id", "payment_type"],
    },
    "order_reviews": {
        "file": "olist_order_reviews_dataset.csv",
        "text": ["review_id", "order_id", "review_comment_title", "review_comment_message"],
        "dates": ["review_creation_date", "review_answer_timestamp"],
    },
    "orders": {
        "file": "olist_orders_dataset.csv",
        "text": ["order_id", "customer_id", "order_status"],
        "dates": ["order_purchase_timestamp", "order_approved_at",
                  "order_delivered_carrier_date", "order_delivered_customer_date",
                  "order_estimated_delivery_date"],
    },
    "products": {
        "file": "olist_products_dataset.csv",
        "text": ["product_id", "product_category_name"],
        "rename": {"product_name_lenght": "product_name_length",
                   "product_description_lenght": "product_description_length"},
    },
    "sellers": {
        "file": "olist_sellers_dataset.csv",
        "text": ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"],
    },
    "category_translation": {
        "file": "product_category_name_translation.csv",
        "text": ["product_category_name", "product_category_name_english"],
        "encoding": "utf-8-sig",
    },
}


def read_table(name: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Le uma tabela da Olist pelo nome curto (ver TABLES)."""
    spec = TABLES[name]
    df = pd.read_csv(
        raw_dir / spec["file"],
        dtype={col: "string" for col in spec.get("text", [])},
        keep_default_na=False,
        na_values=[""],
        encoding=spec.get("encoding", "utf-8"),
    )
    for col in spec.get("dates", []):
        df[col] = pd.to_datetime(df[col], format="%Y-%m-%d %H:%M:%S")
    return df.rename(columns=spec.get("rename", {}))


def read_all(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    return {name: read_table(name, raw_dir) for name in TABLES}


if __name__ == "__main__":
    for name, df in read_all().items():
        print(f"{name:<22} {len(df):>9,} linhas  {df.shape[1]:>2} colunas")
