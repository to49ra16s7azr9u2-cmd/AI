"""Claude に渡すツール定義と、その実行。

- ストアのカタログ検索・詳細・カート操作・好みの記憶はこのプロセス内で実行する（クライアントツール）
- ウェブ検索・ウェブ取得は Anthropic のサーバー側で実行される（サーバーツール）
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .catalog import SORT_KEYS, ProductCatalog
from .config import Settings
from .session import Session

ToolResult = tuple[str, bool]  # (content, is_error)


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        # ストリーミング時に入力を逐次受け取る。サーバー側の入力検証は行われなくなるため、
        # execute() 側で必ず検証する。
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


CLIENT_TOOLS: list[dict[str, Any]] = [
    _tool(
        "search_catalog",
        "当ストアの商品カタログを検索する。ユーザーが商品を探している・比較したい・おすすめを求めているときに使う。"
        "ここで見つかった商品だけがカートに追加できる。キーワードは短い日本語の単語をスペース区切りで"
        "（例: 「イヤホン ノイズキャンセリング」）。ヒットしない場合は言い換えやカテゴリ指定で再検索する。",
        {
            "query": {"type": "string", "description": "検索キーワード（スペース区切り）。空文字ならカテゴリ内の人気順。"},
            "category": {
                "type": "string",
                "description": "カテゴリで絞り込む（家電 / ファッション / 食品 / 日用品 / アウトドア / 美容、またはサブカテゴリ名）。",
            },
            "min_price": {"type": "integer", "description": "最低価格（円・税込）"},
            "max_price": {"type": "integer", "description": "最高価格（円・税込）"},
            "sort": {"type": "string", "enum": list(SORT_KEYS), "description": "並び順。既定は relevance。"},
            "limit": {"type": "integer", "description": "最大件数（1〜20、既定 8）"},
        },
        ["query"],
    ),
    _tool(
        "get_product_details",
        "当ストアの商品の詳細（説明・在庫数・サイズ・カラー）を取得する。複数の商品を比較するときは商品ごとに呼ぶ。",
        {"product_id": {"type": "string", "description": "商品ID（例: EL-001）"}},
        ["product_id"],
    ),
    _tool(
        "add_to_cart",
        "当ストアの商品をカートに追加する。ユーザーが購入・カート追加の意思をはっきり示したときだけ使う。"
        "サイズやカラーがある商品は、ユーザーに確認してから option に指定する。",
        {
            "product_id": {"type": "string", "description": "商品ID"},
            "quantity": {"type": "integer", "description": "数量（1以上）"},
            "option": {"type": "string", "description": "サイズ・カラーなど（例: 「M / ネイビー」）。無ければ空文字。"},
        },
        ["product_id", "quantity"],
    ),
    _tool(
        "remove_from_cart",
        "カートから商品を削除する（quantity を省略すると全数削除）。",
        {
            "product_id": {"type": "string", "description": "商品ID"},
            "quantity": {"type": "integer", "description": "減らす数量。省略時は全数削除。"},
        },
        ["product_id"],
    ),
    _tool(
        "view_cart",
        "現在のカートの中身と合計金額を確認する。",
        {},
        [],
    ),
    _tool(
        "remember_preference",
        "ユーザーの好み・条件を覚えておく（予算、服や靴のサイズ、肌質、好きな/苦手なブランド、家族構成、アレルギー等）。"
        "ユーザーが自分について明かした、今後の提案に役立つ情報だけを保存する。同じ key は上書きされる。",
        {
            "key": {"type": "string", "description": "項目名（例: 予算, 靴のサイズ, 肌質）"},
            "value": {"type": "string", "description": "内容（例: 3万円以内, 26.5cm, 敏感肌）"},
        },
        ["key", "value"],
    ),
]


def server_tools(settings: Settings) -> list[dict[str, Any]]:
    if not settings.enable_web:
        return []
    return [
        {
            "type": "web_search_20260209",
            "name": "web_search",
            "max_uses": settings.web_search_max_uses,
            "user_location": {"type": "approximate", "country": settings.country, "timezone": settings.timezone},
        },
        {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": settings.web_search_max_uses},
    ]


# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------


class ToolInputError(ValueError):
    pass


def _expect(args: dict[str, Any], key: str, typ: type, required: bool) -> Any:
    if key not in args or args[key] is None:
        if required:
            raise ToolInputError(f"'{key}' は必須です")
        return None
    value = args[key]
    # bool は int のサブクラスなので明示的に弾く
    if typ is int and isinstance(value, bool) or not isinstance(value, typ):
        raise ToolInputError(f"'{key}' は {typ.__name__} で指定してください")
    return value


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


class ToolExecutor:
    def __init__(self, catalog: ProductCatalog):
        self.catalog = catalog
        self._handlers: dict[str, Callable[[Session, dict[str, Any]], str]] = {
            "search_catalog": self._search_catalog,
            "get_product_details": self._get_product_details,
            "add_to_cart": self._add_to_cart,
            "remove_from_cart": self._remove_from_cart,
            "view_cart": self._view_cart,
            "remember_preference": self._remember_preference,
        }

    def execute(self, session: Session, name: str, args: Any) -> ToolResult:
        handler = self._handlers.get(name)
        if handler is None:
            return f"不明なツールです: {name}", True
        if not isinstance(args, dict):
            return _json({"INVALID_JSON": _json(args)}), True
        try:
            return handler(session, args), False
        except ToolInputError as e:
            return f"入力エラー: {e}", True

    # --- handlers -----------------------------------------------------------

    def _search_catalog(self, session: Session, args: dict[str, Any]) -> str:
        query = _expect(args, "query", str, True)
        category = _expect(args, "category", str, False) or None
        min_price = _expect(args, "min_price", int, False)
        max_price = _expect(args, "max_price", int, False)
        sort = _expect(args, "sort", str, False) or "relevance"
        if sort not in SORT_KEYS:
            raise ToolInputError(f"sort は {', '.join(SORT_KEYS)} のいずれか")
        limit = _expect(args, "limit", int, False) or 8
        results = self.catalog.search(query, category, min_price, max_price, sort, limit)
        if not results:
            return _json(
                {
                    "results": [],
                    "note": "該当なし。キーワードを短くする・言い換える・カテゴリだけで探すなどを試してください。",
                    "available_categories": self.catalog.categories(),
                }
            )
        return _json({"results": [p.summary() for p in results], "count": len(results)})

    def _get_product_details(self, session: Session, args: dict[str, Any]) -> str:
        pid = _expect(args, "product_id", str, True)
        product = self.catalog.get(pid)
        if product is None:
            raise ToolInputError(f"商品ID {pid} は見つかりません。search_catalog で ID を確認してください")
        return _json(product.detail())

    def _add_to_cart(self, session: Session, args: dict[str, Any]) -> str:
        pid = _expect(args, "product_id", str, True)
        quantity = _expect(args, "quantity", int, True)
        option = (_expect(args, "option", str, False) or "").strip()
        product = self.catalog.get(pid)
        if product is None:
            raise ToolInputError(f"商品ID {pid} は見つかりません")
        if quantity < 1:
            raise ToolInputError("quantity は 1 以上")
        in_cart = sum(i.quantity for i in session.cart.items if i.product.id == product.id)
        if in_cart + quantity > product.stock:
            raise ToolInputError(f"在庫不足です（在庫 {product.stock} 点、カート内 {in_cart} 点）")
        if (product.sizes or product.colors) and not option:
            choices = {"sizes": list(product.sizes), "colors": list(product.colors)}
            raise ToolInputError(f"この商品はサイズ/カラーの指定が必要です。ユーザーに確認してください: {_json(choices)}")
        session.cart.add(product, quantity, option)
        return _json({"ok": True, "added": {"product_id": product.id, "quantity": quantity, "option": option},
                      "cart": session.cart.to_dict()})

    def _remove_from_cart(self, session: Session, args: dict[str, Any]) -> str:
        pid = _expect(args, "product_id", str, True)
        quantity = _expect(args, "quantity", int, False)
        if quantity is not None and quantity < 1:
            raise ToolInputError("quantity は 1 以上")
        if not session.cart.remove(pid, quantity):
            raise ToolInputError(f"{pid} はカートに入っていません")
        return _json({"ok": True, "cart": session.cart.to_dict()})

    def _view_cart(self, session: Session, args: dict[str, Any]) -> str:
        return _json(session.cart.to_dict())

    def _remember_preference(self, session: Session, args: dict[str, Any]) -> str:
        key = _expect(args, "key", str, True).strip()
        value = _expect(args, "value", str, True).strip()
        if not key or not value:
            raise ToolInputError("key と value は空にできません")
        if len(session.preferences) >= 50 and key not in session.preferences:
            raise ToolInputError("保存できる好みの数の上限に達しました")
        session.preferences[key[:50]] = value[:200]
        return _json({"ok": True, "preferences": session.preferences})


CART_TOOLS = frozenset({"add_to_cart", "remove_from_cart"})
