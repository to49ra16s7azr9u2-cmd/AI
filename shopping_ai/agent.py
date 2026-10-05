"""ショッピングエージェント本体: Claude とツールの間を往復するループ。

chat() は非同期ジェネレーターで、UI に流すイベント（dict）を順に返す:
  {"type": "text", "text": "..."}          応答本文の差分
  {"type": "status", "text": "..."}        「検索中…」などの途中経過
  {"type": "cart", "cart": {...}}          カートが変化した
  {"type": "preferences", "preferences": {...}}
  {"type": "error", "text": "..."}         ユーザーに見せるエラー
  {"type": "done", "usage": {...}}         ターン終了
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from .catalog import ProductCatalog
from .config import Settings
from .prompts import build_system_prompt
from .session import Session
from .tools import CART_TOOLS, CLIENT_TOOLS, ToolExecutor, server_tools

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
COMPACTION_BETA = "compact-2026-01-12"
MAX_MODEL_CALLS_PER_TURN = 25
MAX_JSON_RETRIES = 2

_TOOL_STATUS = {
    "search_catalog": "ストアの商品を検索しています…",
    "get_product_details": "商品の詳細を確認しています…",
    "add_to_cart": "カートに追加しています…",
    "remove_from_cart": "カートを更新しています…",
    "view_cart": "カートを確認しています…",
    "remember_preference": "好みを記憶しています…",
}

_INTERNAL_BLOCK_TYPES = {"thinking", "redacted_thinking", "tool_use"}


def _btype(block: Any) -> str | None:
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", None)


def sanitize_for_history(content: list[Any]) -> list[Any]:
    """レスポンス本文を履歴に積める形にする。

    途中でフォールバック（別モデルへの切り替え）が起きた場合、最後の fallback ブロックより前にある
    思考ブロック・tool_use・結果と対になっていない server_tool_use は送り返してはいけない。
    それ以外は手を加えずそのまま返す（履歴は追記のみ）。
    """
    last_fb = max((i for i, b in enumerate(content) if _btype(b) == "fallback"), default=-1)
    if last_fb < 0:
        return list(content)

    def _get(block: Any, attr: str) -> Any:
        return block.get(attr) if isinstance(block, dict) else getattr(block, attr, None)

    paired = {
        _get(b, "tool_use_id")
        for b in content[:last_fb]
        if (_btype(b) or "").endswith("_tool_result")
    }
    kept = []
    for i, b in enumerate(content):
        t = _btype(b)
        if i < last_fb:
            if t in _INTERNAL_BLOCK_TYPES:
                continue
            if t == "server_tool_use" and _get(b, "id") not in paired:
                continue
        kept.append(b)
    return kept


def build_user_content(text: str, images: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
    """images: [{"media_type": "image/png", "data": "<base64>"}]"""
    content: list[dict[str, Any]] = []
    for img in images or []:
        content.append(
            {"type": "image", "source": {"type": "base64", "media_type": img["media_type"], "data": img["data"]}}
        )
    content.append({"type": "text", "text": text or "（画像を送信しました）"})
    return content


class ShoppingAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        client: anthropic.AsyncAnthropic | None = None,
        catalog: ProductCatalog | None = None,
    ):
        self.settings = settings or Settings()
        self.catalog = catalog or ProductCatalog.load(self.settings.catalog_path)
        self.executor = ToolExecutor(self.catalog)
        self._client = client
        self.tools = server_tools(self.settings) + CLIENT_TOOLS

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic()
        return self._client

    # ------------------------------------------------------------------
    def _request_params(self, session: Session) -> dict[str, Any]:
        betas = [FALLBACK_BETA]
        params: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": build_system_prompt(self.catalog, self.settings.timezone),
            "messages": session.messages,
            "tools": self.tools,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.settings.effort},
            # 安全分類器が断った場合は、カテゴリに応じた推奨モデルでサーバー側が自動で再実行する
            "fallbacks": "default",
            # tools → system → messages の最後までを自動でキャッシュ
            "cache_control": {"type": "ephemeral"},
        }
        if self.settings.enable_compaction:
            betas.append(COMPACTION_BETA)
            params["context_management"] = {"edits": [{"type": "compact_20260112"}]}
        params["betas"] = betas
        return params

    async def _stream_once(self, session: Session) -> AsyncIterator[dict[str, Any]]:
        """1 回のモデル呼び出しをストリーミングし、UI イベントを流す。最後に {"type": "_final"} を返す。"""
        async with self.client.beta.messages.stream(**self._request_params(session)) as stream:
            async for event in stream:
                etype = event.type
                if etype == "text":
                    yield {"type": "text", "text": event.text}
                elif etype == "content_block_start":
                    btype = event.content_block.type
                    if btype == "thinking":
                        yield {"type": "status", "text": "考えています…"}
                    elif btype == "server_tool_use":
                        name = event.content_block.name
                        yield {"type": "status", "text": "ウェブで調べています…" if name == "web_search" else "ページを読んでいます…"}
                    elif btype == "tool_use":
                        yield {"type": "status", "text": _TOOL_STATUS.get(event.content_block.name, "処理しています…")}
                    elif btype == "compaction":
                        yield {"type": "status", "text": "これまでの会話を整理しています…"}
                    elif btype == "text":
                        yield {"type": "status", "text": ""}
                elif etype == "content_block_stop":
                    block = getattr(event, "content_block", None)
                    if block is not None and block.type == "server_tool_use":
                        inp = block.input if isinstance(block.input, dict) else {}
                        if block.name == "web_search" and inp.get("query"):
                            yield {"type": "status", "text": f"「{inp['query']}」を検索しています…"}
                        elif block.name == "web_fetch" and inp.get("url"):
                            yield {"type": "status", "text": f"{inp['url']} を読んでいます…"}
            final = await stream.get_final_message()
        yield {"type": "_final", "message": final}

    async def chat(
        self, session: Session, text: str, images: list[dict[str, str]] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """ユーザーの 1 発言を処理する。同じセッションの同時実行は session.lock で防ぐこと。"""
        start_len = len(session.messages)
        session.messages.append({"role": "user", "content": build_user_content(text, images)})
        usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "model_calls": 0}
        json_retries = 0

        try:
            while usage["model_calls"] < MAX_MODEL_CALLS_PER_TURN:
                usage["model_calls"] += 1
                final = None
                try:
                    async for ev in self._stream_once(session):
                        if ev["type"] == "_final":
                            final = ev["message"]
                        else:
                            yield ev
                except ValueError:
                    # eager_input_streaming 中にツール入力の JSON が完全に壊れていた場合。ターンを再試行する。
                    json_retries += 1
                    if json_retries > MAX_JSON_RETRIES:
                        raise
                    log.warning("tool input JSON could not be parsed; retrying model call")
                    continue
                json_retries = 0
                assert final is not None
                self._add_usage(usage, final)

                if final.stop_reason == "refusal":
                    # 断られたターンは履歴に残さない（末尾を巻き戻すだけで、過去のターンは変えない）
                    del session.messages[start_len:]
                    yield {"type": "error", "text": "申し訳ありません、この内容にはお答えできません。別の聞き方でお試しください。"}
                    break

                session.messages.append({"role": "assistant", "content": sanitize_for_history(final.content)})

                if final.stop_reason == "pause_turn":
                    # サーバーツール（ウェブ検索など）が長引いて一時停止した。そのまま続きを依頼する。
                    continue

                tool_uses = [b for b in final.content if b.type == "tool_use"]
                if final.stop_reason == "max_tokens":
                    if tool_uses:
                        del session.messages[start_len:]
                    yield {"type": "error", "text": "応答が長くなりすぎたため途中で止まりました。「続けて」と送ると続きを書きます。"}
                    break
                if final.stop_reason != "tool_use" or not tool_uses:
                    break

                results = []
                for block in tool_uses:
                    content, is_error = self.executor.execute(session, block.name, block.input)
                    log.info("tool %s(%s) -> error=%s", block.name, block.input, is_error)
                    result: dict[str, Any] = {"type": "tool_result", "tool_use_id": block.id, "content": content}
                    if is_error:
                        result["is_error"] = True
                    results.append(result)
                    if not is_error and block.name in CART_TOOLS:
                        yield {"type": "cart", "cart": session.cart.to_dict()}
                    if not is_error and block.name == "remember_preference":
                        yield {"type": "preferences", "preferences": dict(session.preferences)}
                session.messages.append({"role": "user", "content": results})
                # 本文の区切り（ツール前後のテキストがくっつかないように）
                yield {"type": "text", "text": "\n\n"}
            else:
                yield {"type": "error", "text": "処理が長くなりすぎたため中断しました。質問を分けてお試しください。"}
        except anthropic.APIError as e:
            del session.messages[start_len:]
            log.exception("Claude API error")
            yield {"type": "error", "text": _friendly_api_error(e)}
        except Exception:
            del session.messages[start_len:]
            log.exception("unexpected error in agent loop")
            yield {"type": "error", "text": "予期しないエラーが発生しました。もう一度お試しください。"}

        yield {"type": "done", "usage": usage}

    @staticmethod
    def _add_usage(acc: dict[str, int], message: Any) -> None:
        u = getattr(message, "usage", None)
        if u is None:
            return
        acc["input_tokens"] += getattr(u, "input_tokens", 0) or 0
        acc["output_tokens"] += getattr(u, "output_tokens", 0) or 0
        acc["cache_read_input_tokens"] += getattr(u, "cache_read_input_tokens", 0) or 0


def _friendly_api_error(e: anthropic.APIError) -> str:
    if isinstance(e, anthropic.AuthenticationError):
        return "API キーが無効です。ANTHROPIC_API_KEY を確認してください。"
    if isinstance(e, anthropic.PermissionDeniedError):
        return "この API キーではこの機能を利用できません。"
    if isinstance(e, anthropic.RateLimitError):
        return "混み合っています。少し時間をおいてお試しください。"
    if isinstance(e, anthropic.BadRequestError):
        return f"リクエストエラー: {e.message}"
    if isinstance(e, anthropic.APIStatusError) and e.status_code >= 500:
        return "AI サーバーで一時的な問題が発生しています。少し待ってからお試しください。"
    if isinstance(e, anthropic.APIConnectionError):
        return "AI サーバーに接続できませんでした。ネットワークを確認してください。"
    return "AI の呼び出しでエラーが発生しました。"
