"""1 人のユーザーとの会話状態（履歴・カート・好み）。"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from .catalog import Product


@dataclass
class CartItem:
    product: Product
    quantity: int
    option: str = ""  # サイズ・色など

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product.id,
            "name": self.product.name,
            "unit_price_jpy": self.product.price,
            "quantity": self.quantity,
            "option": self.option,
            "subtotal_jpy": self.product.price * self.quantity,
        }


@dataclass
class Cart:
    items: list[CartItem] = field(default_factory=list)

    def add(self, product: Product, quantity: int, option: str = "") -> CartItem:
        for item in self.items:
            if item.product.id == product.id and item.option == option:
                item.quantity += quantity
                return item
        item = CartItem(product, quantity, option)
        self.items.append(item)
        return item

    def remove(self, product_id: str, quantity: int | None = None) -> bool:
        """quantity が None なら該当商品をすべて削除。"""
        pid = product_id.strip().upper()
        found = False
        for item in list(self.items):
            if item.product.id != pid:
                continue
            found = True
            if quantity is None or quantity >= item.quantity:
                self.items.remove(item)
            else:
                item.quantity -= quantity
            if quantity is not None:
                break
        return found

    @property
    def total(self) -> int:
        return sum(i.product.price * i.quantity for i in self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [i.to_dict() for i in self.items],
            "item_count": sum(i.quantity for i in self.items),
            "total_jpy": self.total,
        }


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Claude API に送る会話履歴（追記のみ。過去のターンは書き換えない）
    messages: list[dict[str, Any]] = field(default_factory=list)
    cart: Cart = field(default_factory=Cart)
    # 予算・サイズ・好きなブランド・アレルギーなど、会話から覚えた好み
    preferences: dict[str, str] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def snapshot(self) -> dict[str, Any]:
        return {"session_id": self.id, "cart": self.cart.to_dict(), "preferences": dict(self.preferences)}


class SessionStore:
    """メモリ上のセッション管理。本番では Redis / DB に置き換える。"""

    def __init__(self, max_sessions: int = 1000):
        self._sessions: dict[str, Session] = {}
        self._max = max_sessions

    def get_or_create(self, session_id: str | None) -> Session:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        if len(self._sessions) >= self._max:
            # 最も古いセッションを捨てる（dict は挿入順）
            self._sessions.pop(next(iter(self._sessions)))
        session = Session(id=session_id) if session_id else Session()
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def reset(self, session_id: str) -> Session:
        self._sessions.pop(session_id, None)
        return self.get_or_create(session_id)
