"""環境変数から読み込む設定値。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    model: str = field(default_factory=lambda: os.environ.get("SHOPPING_AI_MODEL", "claude-opus-5-5"))
    # low | medium | high | xhigh | max。買い物の調査・比較は品質重視で high を既定にする。
    effort: str = field(default_factory=lambda: os.environ.get("SHOPPING_AI_EFFORT", "high"))
    max_tokens: int = field(default_factory=lambda: int(os.environ.get("SHOPPING_AI_MAX_TOKENS", "64000")))
    # ウェブ検索・取得（最新の価格・レビュー・在庫を調べる）を使うか
    enable_web: bool = field(
        default_factory=lambda: os.environ.get("SHOPPING_AI_ENABLE_WEB", "1") not in ("0", "false", "False")
    )
    web_search_max_uses: int = field(default_factory=lambda: int(os.environ.get("SHOPPING_AI_WEB_MAX_USES", "8")))
    # 長い会話をサーバー側で自動要約（compaction）するか
    enable_compaction: bool = field(
        default_factory=lambda: os.environ.get("SHOPPING_AI_COMPACTION", "1") not in ("0", "false", "False")
    )
    catalog_path: Path = field(
        default_factory=lambda: Path(os.environ.get("SHOPPING_AI_CATALOG", PACKAGE_DIR / "data" / "catalog.json"))
    )
    country: str = field(default_factory=lambda: os.environ.get("SHOPPING_AI_COUNTRY", "JP"))
    timezone: str = field(default_factory=lambda: os.environ.get("SHOPPING_AI_TIMEZONE", "Asia/Tokyo"))
