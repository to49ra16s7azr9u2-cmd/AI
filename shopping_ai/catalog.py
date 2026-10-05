"""ストアの商品カタログ（デモ用 JSON）の読み込みと検索。

本番では、この ProductCatalog を自社 EC の商品 API / DB を叩く実装に差し替える。
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SPLIT_RE = re.compile(r"[\s、。,.!?！？・/]+")
# 「〜が欲しい」「〜好き」など、検索語の末尾についても意味の薄い語
_SUFFIX_RE = re.compile(r"(が欲しい|がほしい|が好き|好き|向け|用|系|グッズ|関連|など|とか|っぽい|みたい)$")

# 言い換え・略語・用途 → カタログ中の語。キーも値も normalize() 後の形で照合する。
SYNONYMS: dict[str, tuple[str, ...]] = {
    "ノイキャン": ("ノイズキャンセリング",),
    "anc": ("ノイズキャンセリング",),
    "ワイヤレス": ("bluetooth", "ワイヤレス"),
    "ヘッドフォン": ("ヘッドホン",),
    "イヤフォン": ("イヤホン",),
    "プレゼント": ("ギフト",),
    "贈り物": ("ギフト",),
    "誕生日": ("ギフト",),
    "お祝い": ("ギフト", "内祝い"),
    "お返し": ("ギフト", "お返し", "内祝い"),
    "手土産": ("ギフト", "個包装"),
    "母": ("ギフト",), "父": ("ギフト",), "母の日": ("ギフト",), "父の日": ("ギフト",),
    "彼女": ("ギフト",), "彼氏": ("ギフト",), "友達": ("ギフト",), "上司": ("ギフト",),
    "クリスマス": ("ギフト",), "退職祝い": ("ギフト", "名入れ可"), "出産祝い": ("今治", "ギフト"),
    "名入れ": ("名入れ可",),
    "子供": ("キッズ", "ベビー"), "子ども": ("キッズ", "ベビー"), "こども": ("キッズ", "ベビー"),
    "赤ちゃん": ("ベビー", "出産祝い"), "赤ちゃん用": ("ベビー",), "幼児": ("キッズ", "知育"),
    "おもちゃ": ("おもちゃ", "知育"), "ペット": ("ペット", "犬", "猫"),
    "文房具": ("文具",), "ペン": ("ボールペン", "筆記具"),
    "敬老": ("敬老の日", "ギフト"), "祖父": ("ギフト", "敬老の日"), "祖母": ("ギフト", "敬老の日"),
    "おじいちゃん": ("ギフト", "敬老の日"), "おばあちゃん": ("ギフト", "敬老の日"),
    "就活": ("就職祝い", "ビジネス"), "新社会人": ("就職祝い", "ビジネス"),
    "防災": ("防災", "ポータブル電源", "モバイルバッテリー"),
    "停電": ("防災", "ポータブル電源"),
    "肌荒れ": ("敏感肌",), "乾燥": ("乾燥肌", "保湿"), "アトピー": ("敏感肌",),
    "uv": ("日焼け止め", "uvカット"), "紫外線": ("日焼け止め", "uvカット"),
    "日焼け": ("日焼け止め",),
    "ダイエット": ("ダイエット", "ヘルシー", "プロテイン"),
    "筋トレ": ("筋トレ", "プロテイン"),
    "ジョギング": ("ランニング",), "マラソン": ("ランニング", "マラソン"),
    "スニーカー": ("シューズ",), "靴": ("シューズ",),
    "コート": ("アウター", "コート"), "ジャケット": ("アウター", "ジャケット"),
    "寒さ対策": ("秋冬", "ダウン", "マフラー"), "冬": ("秋冬",), "秋": ("秋冬",),
    "財布": ("財布",), "ウォレット": ("財布",),
    "パソコン": ("ノートpc", "パソコン"), "pc": ("ノートpc", "パソコン"),
    "ノートパソコン": ("ノートpc",),
    "充電器": ("モバイルバッテリー", "充電"),
    "スマートウォッチ": ("スマートウォッチ", "ウェアラブル"),
    "掃除": ("掃除機", "洗剤", "掃除"),
    "時短": ("時短", "ロボット掃除機", "ノンフライヤー"),
    "一人暮らし": ("一人暮らし", "コンパクト", "軽量"),
    "コスパ": ("コスパ",), "安い": ("コスパ",),
    "在宅": ("在宅勤務",), "テレワーク": ("在宅勤務",),
    "通勤": ("通勤",), "旅行": ("旅行", "パッカブル", "軽量"),
    "キャンプ": ("キャンプ", "アウトドア"), "アウトドア": ("アウトドア", "キャンプ"),
    "車中泊": ("車中泊", "ポータブル電源"),
    "寝具": ("寝具", "まくら"), "枕": ("まくら",), "肩こり": ("肩こり", "まくら"),
    "お菓子": ("お菓子", "焼き菓子"), "スイーツ": ("スイーツ", "焼き菓子"),
    "コーヒー": ("コーヒー",), "珈琲": ("コーヒー",),
    "お茶": ("お茶",), "水筒": ("タンブラー", "保温保冷"),
}


def normalize(text: str) -> str:
    """全角/半角・大文字/小文字・ひらがな/カタカナの違いを吸収する。"""
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(chr(ord(ch) + 0x60) if "ぁ" <= ch <= "ゖ" else ch for ch in text)


def _bigrams(text: str) -> set[str]:
    return {text[i : i + 2] for i in range(len(text) - 1)}


# 照合は normalize() 後の文字列同士で行うので、辞書と接尾辞も正規化しておく
_SYN_N: dict[str, tuple[str, ...]] = {normalize(k): tuple(normalize(x) for x in v) for k, v in SYNONYMS.items()}
_SUFFIX_N = re.compile(
    "(" + "|".join(re.escape(normalize(x)) for x in _SUFFIX_RE.pattern[1:-2].split("|")) + ")$"
)


_AGE_RE = re.compile(r"^\d{1,2}(歳|才|サイ)$")


def _expand(term: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """正規化済みの検索語 1 つを (そのままの語, 言い換え) の候補に展開する。"""
    direct = [term]
    stripped = _SUFFIX_N.sub("", term)
    if stripped and stripped != term:
        direct.append(stripped)
    related: list[str] = []
    for t in direct:
        related.extend(_SYN_N.get(t, ()))
        # 「ワイヤレスイヤフォン」のような複合語: 2 文字以上の辞書語を含んでいれば、その言い換えも候補にする
        for key, values in _SYN_N.items():
            if len(key) >= 2 and key in t and key != t:
                related.extend(values)
                related.extend(t.replace(key, v) for v in values)
        if _AGE_RE.match(t):
            related.extend(normalize(x) for x in ("キッズ", "ベビー", "知育", "歳から"))
    direct = list(dict.fromkeys(direct))
    return tuple(direct), tuple(x for x in dict.fromkeys(related) if x not in direct)


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
        return normalize(
            " ".join([self.name, self.brand, self.category, self.subcategory, self.description, *self.tags, *self.colors])
        )

    def _strong_text(self) -> str:
        return normalize(" ".join([self.name, self.subcategory, *self.tags]))


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
        terms = [_expand(t) for t in _SPLIT_RE.split(normalize(query)) if t]
        scored: list[tuple[float, Product]] = []
        fuzzy_hits: list[tuple[float, Product]] = []
        for p in self._products:
            if category and normalize(category) not in (normalize(p.category), normalize(p.subcategory)):
                continue
            if min_price is not None and p.price < min_price:
                continue
            if max_price is not None and p.price > max_price:
                continue
            if terms:
                exact, fuzzy = self._score(p, terms)
                if exact > 0:
                    scored.append((exact + p.rating * 0.05, p))
                elif fuzzy > 0:
                    fuzzy_hits.append((fuzzy + p.rating * 0.05, p))
            else:
                scored.append((p.rating, p))
        # 完全一致が 1 件もないときだけ、表記ゆれを許したあいまい一致の結果を使う
        if not scored:
            scored = fuzzy_hits

        if sort == "price_asc":
            scored.sort(key=lambda sp: sp[1].price)
        elif sort == "price_desc":
            scored.sort(key=lambda sp: -sp[1].price)
        elif sort == "rating":
            scored.sort(key=lambda sp: (-sp[1].rating, -sp[1].reviews))
        else:
            scored.sort(key=lambda sp: -sp[0])
        return [p for _, p in scored[: max(1, min(limit, 20))]]

    @staticmethod
    def _score(p: Product, terms: list[tuple[tuple[str, ...], tuple[str, ...]]]) -> tuple[float, float]:
        """(完全一致スコア, あいまい一致スコア) を返す。"""
        hay, strong = p._haystack(), p._strong_text()
        hay_grams = _bigrams(hay)
        exact = fuzzy = 0.0
        for direct, related in terms:
            # 書かれたままの語の一致を、言い換えでの一致より重く見る
            if any(v in strong for v in direct):
                exact += 2.0
            elif any(v in hay for v in direct):
                exact += 1.5
            elif any(v in strong for v in related):
                exact += 1.2
            elif any(v in hay for v in related):
                exact += 0.8
            else:
                # 表記ゆれ・打ち間違い対策: 3 文字以上の語は文字バイグラムの重なりで部分一致を見る
                best = max(
                    (len(_bigrams(v) & hay_grams) / len(_bigrams(v)) for v in direct + related if len(v) >= 3),
                    default=0.0,
                )
                if best >= 0.6:
                    fuzzy += best
        return exact, fuzzy
