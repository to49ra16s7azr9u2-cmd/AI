import json

import pytest

from shopping_ai.catalog import ProductCatalog
from shopping_ai.config import Settings
from shopping_ai.session import Session
from shopping_ai.tools import CLIENT_TOOLS, ToolExecutor


@pytest.fixture
def catalog():
    return ProductCatalog.load(Settings().catalog_path)


@pytest.fixture
def ex(catalog):
    return ToolExecutor(catalog)


def test_catalog_loads(catalog):
    assert len(catalog) >= 30
    assert "家電" in catalog.categories()


def test_search_keyword_and_price(catalog):
    res = catalog.search("イヤホン ノイズキャンセリング", max_price=30000)
    assert res and res[0].id == "EL-001"
    assert all(p.price <= 30000 for p in res)


def test_search_sort_price(catalog):
    res = catalog.search("", category="家電", sort="price_asc", limit=20)
    prices = [p.price for p in res]
    assert prices == sorted(prices)


def test_search_no_hit_returns_note(ex):
    out, err = ex.execute(Session(), "search_catalog", {"query": "宇宙船"})
    assert not err
    data = json.loads(out)
    assert data["results"] == [] and "available_categories" in data


def test_tool_schemas_are_well_formed():
    names = set()
    for t in CLIENT_TOOLS:
        assert t["input_schema"]["type"] == "object"
        assert set(t["input_schema"]["required"]) <= set(t["input_schema"]["properties"])
        names.add(t["name"])
    assert len(names) == len(CLIENT_TOOLS)


def test_add_to_cart_requires_option_for_sized_items(ex):
    s = Session()
    out, err = ex.execute(s, "add_to_cart", {"product_id": "FA-004", "quantity": 1})
    assert err and "サイズ" in out
    out, err = ex.execute(s, "add_to_cart", {"product_id": "FA-004", "quantity": 1, "option": "26.5 / ブラック"})
    assert not err
    assert s.cart.total == 14300


def test_cart_add_merge_remove(ex):
    s = Session()
    ex.execute(s, "add_to_cart", {"product_id": "el-007", "quantity": 1})
    ex.execute(s, "add_to_cart", {"product_id": "EL-007", "quantity": 2})
    assert s.cart.to_dict()["item_count"] == 3
    out, err = ex.execute(s, "remove_from_cart", {"product_id": "EL-007", "quantity": 1})
    assert not err and s.cart.to_dict()["item_count"] == 2
    ex.execute(s, "remove_from_cart", {"product_id": "EL-007"})
    assert s.cart.items == []
    _, err = ex.execute(s, "remove_from_cart", {"product_id": "EL-007"})
    assert err


def test_stock_limit(ex):
    s = Session()
    out, err = ex.execute(s, "add_to_cart", {"product_id": "OD-003", "quantity": 10})
    assert err and "在庫" in out


@pytest.mark.parametrize(
    "args",
    [{"product_id": "EL-001"}, {"product_id": "EL-001", "quantity": "2"}, {"product_id": "EL-001", "quantity": True},
     {"product_id": "NOPE", "quantity": 1}, {"product_id": "EL-001", "quantity": 0}, "not a dict"],
)
def test_invalid_inputs_are_errors(ex, args):
    _, err = ex.execute(Session(), "add_to_cart", args)
    assert err


def test_remember_preference(ex):
    s = Session()
    _, err = ex.execute(s, "remember_preference", {"key": "予算", "value": "3万円以内"})
    assert not err and s.preferences == {"予算": "3万円以内"}


def test_unknown_tool(ex):
    _, err = ex.execute(Session(), "launch_rocket", {})
    assert err
