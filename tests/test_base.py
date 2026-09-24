"""Cardinalidade das juncoes e regras de agregacao da base de avaliacoes."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.base import build_base

T = pd.Timestamp


def make_tables() -> dict[str, pd.DataFrame]:
    """Cenarios pequenos que cobrem cada regra da base.

    r1: uma avaliacao para dois pedidos (o1 entregue, o2 nao entregue)
    r2 e r3: duas avaliacoes do mesmo pedido o3, entregue com atraso
    r4: pedido o4 cancelado e sem itens
    r5: pedido o5 entregue no dia da estimativa, com empate de valor entre categorias
    """
    reviews = pd.DataFrame({
        "review_id": ["r1", "r1", "r2", "r3", "r4", "r5"],
        "order_id": ["o1", "o2", "o3", "o3", "o4", "o5"],
        "review_score": [3, 3, 1, 2, 1, 5],
        "review_comment_title": [None, None, "atraso", None, None, None],
        "review_comment_message": ["chegou metade", "chegou metade", "demorou", "   ", None,
                                   "otimo"],
        "review_creation_date": [T("2018-01-10")] * 2 + [T("2018-02-20")] * 2
        + [T("2018-03-05"), T("2018-04-10")],
        "review_answer_timestamp": [T("2018-01-11 10:00")] * 2
        + [T("2018-02-21 09:00"), T("2018-02-25 09:00"), T("2018-03-06"), T("2018-04-08")],
    })
    orders = pd.DataFrame({
        "order_id": ["o1", "o2", "o3", "o4", "o5"],
        "customer_id": ["c1", "c2", "c3", "c4", "c5"],
        "order_status": ["delivered", "shipped", "delivered", "canceled", "delivered"],
        "order_purchase_timestamp": [T("2018-01-01"), T("2018-01-02"), T("2018-02-01"),
                                     T("2018-03-01"), T("2018-04-01 08:00")],
        "order_approved_at": [T("2018-01-01")] * 2 + [T("2018-02-01"), pd.NaT,
                                                      T("2018-04-01")],
        "order_delivered_carrier_date": [T("2018-01-03"), T("2018-01-04"), T("2018-02-05"),
                                         pd.NaT, T("2018-04-02")],
        "order_delivered_customer_date": [T("2018-01-08"), pd.NaT, T("2018-02-19 15:00"),
                                          pd.NaT, T("2018-04-09 18:00")],
        "order_estimated_delivery_date": [T("2018-01-15"), T("2018-01-20"), T("2018-02-15"),
                                          T("2018-03-20"), T("2018-04-09")],
    })
    customers = pd.DataFrame({
        "customer_id": ["c1", "c2", "c3", "c4", "c5"],
        "customer_unique_id": ["u1", "u1", "u3", "u4", "u5"],
        "customer_state": ["SP", "SP", "BA", "RS", "AM"],
    })
    items = pd.DataFrame({
        "order_id": ["o1", "o2", "o2", "o3", "o5", "o5"],
        "order_item_id": [1, 1, 2, 1, 1, 2],
        "product_id": ["p_cama", "p_relogio", "p_relogio", "p_cama", "p_beleza", "p_esporte"],
        "seller_id": ["s1", "s2", "s2", "s1", "s3", "s4"],
        "price": [50.0, 30.0, 30.0, 80.0, 40.0, 40.0],
        "freight_value": [10.0, 5.0, 5.0, 12.0, 8.0, 8.0],
    })
    products = pd.DataFrame({
        "product_id": ["p_cama", "p_relogio", "p_beleza", "p_esporte"],
        "product_category_name": ["cama_mesa_banho", "relogios_presentes", "beleza_saude",
                                  "esporte_lazer"],
    })
    return {"order_reviews": reviews, "orders": orders, "order_items": items,
            "products": products, "customers": customers}


@pytest.fixture(scope="module")
def base() -> pd.DataFrame:
    return build_base(make_tables()).set_index("review_id")


def test_one_row_per_review(base):
    assert base.index.is_unique
    assert sorted(base.index) == ["r1", "r2", "r3", "r4", "r5"]


def test_review_covering_two_orders(base):
    r1 = base.loc["r1"]
    assert r1["n_orders"] == 2
    assert sorted(r1["order_ids"]) == ["o1", "o2"]
    assert r1["n_items"] == 3
    assert r1["items_value"] == pytest.approx(110.0)
    # relogios somam 60 contra 50 de cama: a categoria principal segue o valor
    assert r1["main_category"] == "relogios_presentes"
    assert r1["categories"] == ["cama_mesa_banho", "relogios_presentes"]
    # um dos pedidos nao foi entregue: a avaliacao conta como nao entregue
    assert r1["order_status"] == "shipped"
    assert pd.isna(r1["order_delivered_customer_date"])
    assert r1["delivery_status"] == "nao entregue"
    assert r1["order_purchase_timestamp"] == T("2018-01-01")


def test_order_with_two_reviews_keeps_both(base):
    assert base.loc["r2", "order_ids"] == ["o3"] and base.loc["r3", "order_ids"] == ["o3"]
    assert base.loc["r2", "order_has_multiple_reviews"]
    assert base.loc["r3", "order_has_multiple_reviews"]
    assert not base.loc["r5", "order_has_multiple_reviews"]


def test_late_delivery(base):
    r2 = base.loc["r2"]
    assert r2["delay_days"] == 4
    assert bool(r2["is_late"]) is True
    assert r2["delivery_status"] == "atrasada"


def test_delivery_on_estimated_day_is_on_time(base):
    r5 = base.loc["r5"]
    assert r5["delay_days"] == 0
    assert bool(r5["is_late"]) is False
    assert r5["delivery_status"] == "no prazo"


def test_canceled_order_without_items(base):
    r4 = base.loc["r4"]
    assert r4["delivery_status"] == "cancelada"
    assert pd.isna(r4["is_late"])
    assert r4["n_items"] == 0 and r4["n_categories"] == 0
    assert pd.isna(r4["main_category"])
    assert list(r4["categories"]) == []


def test_category_tie_breaks_alphabetically(base):
    assert base.loc["r5", "main_category"] == "beleza_saude"


def test_answered_before_delivery(base):
    assert bool(base.loc["r5", "answered_before_delivery"]) is True
    assert bool(base.loc["r2", "answered_before_delivery"]) is False
    assert bool(base.loc["r1", "answered_before_delivery"]) is True


def test_comment_flags_ignore_blank_text(base):
    assert bool(base.loc["r2", "has_comment"]) and bool(base.loc["r2", "has_title"])
    assert not bool(base.loc["r3", "has_comment"])
    assert not bool(base.loc["r4", "has_comment"])


def test_region_and_period(base):
    assert base.loc["r5", "customer_region"] == "Norte"
    assert base.loc["r2", "purchase_month"] == "2018-02"
    assert base.loc["r2", "purchase_year"] == 2018


PROCESSED = Path("data/processed/avaliacoes.parquet")


@pytest.mark.skipif(not PROCESSED.exists(), reason="base processada ausente")
def test_processed_base_contract():
    df = pd.read_parquet(PROCESSED)
    assert df["review_id"].is_unique
    assert (df["n_orders"] >= 1).all()
    assert set(df["delivery_status"]) <= {"no prazo", "atrasada", "nao entregue", "cancelada"}
    assert df["is_late"].isna().eq(~df["delivered"]).all()
    assert df["review_score"].between(1, 5).all()
    assert (df["n_categories"] == df["categories"].apply(len)).all()
