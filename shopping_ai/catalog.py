"""ストアの商品カタログ（デモ用 JSON）の読み込みと検索。

本番では、この ProductCatalog を自社 EC の商品 API / DB を叩く実装に差し替える。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SPLIT_RE = re.compile(r"[\s　、,・/]+")


@dataclass(frozen=True)
class Product:
    id: str
    name: str
    brand: str
    category: str
    subcategory: str
    price: int
    rating: float
    reviews: int
    stock: int
    tags: tuple[str, ...]
    description: str
    sizes: tuple[str, ...] = ()
    colors: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Product":
        return cls(
            id=d["id"],
            name=d["name"],
            brand=d["brand"],
            category=d["category"],
            subcategory=d.get("subcategory", ""),
            price=int(d["price"]),
            rating=float(d.get("rating", 0)),
            reviews=int(d.get("reviews", 0)),
            stock=int(d.get("stock", 0)),
            tags=tuple(d.get("tags", ())),
            description=d.get("description", ""),
            sizes=tuple(d.get("sizes", ())),
            colors=tuple(d.get("colors", ())),
        )

    def summary(self) -> dict[str, Any]:
        """検索結果用の要約（トークン節約のため説明文は省く）。"""
        out: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "brand": self.brand,
            "category": f"{self.category} > {self.subcategory}" if self.subcategory else self.category,
            "price_jpy": self.price,
            "rating": self.rating,
            "reviews": self.reviews,
            "in_stock": self.stock > 0,
            "tags": list(self.tags),
        }
        return out

    def detail(self) -> dict[str, Any]:
        out = self.summary()
        out["description"] = self.description
        out["stock"] = self.stock
        if self.sizes:
            out["sizes"] = list(self.sizes)
        if self.colors:
            out["colors"] = list(self.colors)
        return out

    def _haystack(self) -> str:
        return " ".join(
            [self.name, self.brand, self.category, self.subcategory, self.description, *self.tags, *self.colors]
        ).lower()


SORT_KEYS = ("relevance", "price_asc", "price_desc", "rating")


class ProductCatalog:
    def __init__(self, products: list[Product]):
        self._products = products
        self._by_id = {p.id: p for p in products}

    @classmethod
    def load(cls, path: Path) -> "ProductCatalog":
        with open(path, encoding="utf-8") as f:
            return cls([Product.from_dict(d) for d in json.load(f)])

    def __len__(self) -> int:
        return len(self._products)

    def get(self, product_id: str) -> Product | None:
        return self._by_id.get(product_id.strip().upper())

    def categories(self) -> list[str]:
        return sorted({p.category for p in self._products})

    def search(
        self,
        query: str = "",
        category: str | None = None,
        min_price: int | None = None,
        max_price: int | None = None,
        sort: str = "relevance",
        limit: int = 8,
    ) -> list[Product]:
        terms = [t for t in _SPLIT_RE.split(query.lower()) if t]
        scored: list[tuple[float, Product]] = []
        for p in self._products:
            if category and category not in (p.category, p.subcategory):
                continue
            if min_price is not None and p.price < min_price:
                continue
            if max_price is not None and p.price > max_price:
                continue
            hay = p._haystack()
            if terms:
                hits = sum(1 for t in terms if t in hay)
                if hits == 0:
                    continue
                # 名前・タグでの一致を重く見る
                strong = sum(1 for t in terms if t in p.name.lower() or any(t in tag.lower() for tag in p.tags))
                score = hits + strong * 0.5 + p.rating * 0.05
            else:
                score = p.rating
            scored.append((score, p))

        if sort == "price_asc":
            scored.sort(key=lambda sp: sp[1].price)
        elif sort == "price_desc":
            scored.sort(key=lambda sp: -sp[1].price)
        elif sort == "rating":
            scored.sort(key=lambda sp: (-sp[1].rating, -sp[1].reviews))
        else:
            scored.sort(key=lambda sp: -sp[0])
        return [p for _, p in scored[: max(1, min(limit, 20))]]
