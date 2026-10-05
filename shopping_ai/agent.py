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

import json
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
THINKING_UPDATES_BETA = "thinking-display-updates-2026-08-18"
# display="updates" のとき、出力が途中で打ち切られた場合に返る定型文（状態表示には出さない）
_INTERRUPTED_SENTINEL = "This part of the response was interrupted before it finished."
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


def build_user_content(
    text: str, images: list[dict[str, str]] | None = None, notes: list[str] | None = None
) -> list[dict[str, Any]]:
    """images: [{"media_type": "image/png", "data": "<base64>"}]
    notes: ユーザーが画面のボタンで行った操作の記録。モデルが状況を把握できるよう本文の前に付ける。
    """
    content: list[dict[str, Any]] = []
    if notes:
        content.append({"type": "text", "text": "[画面操作メモ] " + " / ".join(notes)})
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
        betas = [FALLBACK_BETA, THINKING_UPDATES_BETA]
        params: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": build_system_prompt(self.catalog, self.settings.timezone),
            "messages": session.messages,
            "tools": self.tools,
            # "updates": 推論そのものは返さず、ツール呼び出しの合間の短い進捗メモだけを返す（状態表示に使う）
            "thinking": {"type": "adaptive", "display": "updates"},
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
        progress = ""
        async with self.client.beta.messages.stream(**self._request_params(session)) as stream:
            async for event in stream:
                etype = event.type
                if etype == "content_block_delta" and getattr(event.delta, "type", None) == "thinking_delta":
                    progress += event.delta.thinking or ""
                    note = progress.strip()
                    if note and not note.startswith(_INTERRUPTED_SENTINEL[:20]):
                        yield {"type": "status", "text": _shorten(note)}
                elif etype == "text":
                    yield {"type": "text", "text": event.text}
                elif etype == "content_block_start":
                    btype = event.content_block.type
                    if btype == "thinking":
                        progress = ""
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
        notes, session.pending_notes = session.pending_notes, []
        session.messages.append({"role": "user", "content": build_user_content(text, images, notes)})
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
                    _rollback(session, start_len, notes)
                    yield {"type": "error", "text": "申し訳ありません、この内容にはお答えできません。別の聞き方でお試しください。"}
                    break

                session.messages.append({"role": "assistant", "content": sanitize_for_history(final.content)})

                if final.stop_reason == "pause_turn":
                    # サーバーツール（ウェブ検索など）が長引いて一時停止した。そのまま続きを依頼する。
                    continue

                tool_uses = [b for b in final.content if b.type == "tool_use"]
                if final.stop_reason == "max_tokens":
                    if tool_uses:
                        _rollback(session, start_len, notes)
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
            _rollback(session, start_len, notes)
            log.exception("Claude API error")
            yield {"type": "error", "text": _friendly_api_error(e)}
        except Exception:
            _rollback(session, start_len, notes)
            log.exception("unexpected error in agent loop")
            yield {"type": "error", "text": "予期しないエラーが発生しました。もう一度お試しください。"}

        yield {"type": "done", "usage": usage}

    async def suggest_followups(self, session: Session) -> list[str]:
        """直近のやり取りから、ユーザーが次に送りそうなメッセージ候補を 3 つ作る（画面の候補ボタン用）。

        会話履歴本体には一切追加しない別リクエスト。失敗しても会話には影響させない。
        """
        last_user = last_ai = ""
        for m in reversed(session.messages):
            text = _message_text(m)
            if not text:
                continue  # ツール結果だけのターンなど
            if m["role"] == "assistant" and not last_ai:
                last_ai = text
            elif m["role"] == "user" and last_ai:
                last_user = text
                break
        if not last_ai:
            return []
        try:
            resp = await self.client.messages.create(
                model=self.settings.model,
                max_tokens=2000,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": _SUGGEST_SCHEMA}},
                messages=[{"role": "user", "content": _SUGGEST_PROMPT.format(user=last_user[:2000], ai=last_ai[:4000])}],
            )
            if resp.stop_reason != "end_turn":
                return []
            raw = next((b.text for b in resp.content if b.type == "text"), "")
            items = json.loads(raw).get("suggestions", [])
            return [str(x).strip()[:60] for x in items if str(x).strip()][:3]
        except (anthropic.APIError, ValueError, KeyError, TypeError):
            log.warning("follow-up suggestion failed", exc_info=True)
            return []

    @staticmethod
    def _add_usage(acc: dict[str, int], message: Any) -> None:
        u = getattr(message, "usage", None)
        if u is None:
            return
        acc["input_tokens"] += getattr(u, "input_tokens", 0) or 0
        acc["output_tokens"] += getattr(u, "output_tokens", 0) or 0
        acc["cache_read_input_tokens"] += getattr(u, "cache_read_input_tokens", 0) or 0


_SUGGEST_SCHEMA = {
    "type": "object",
    "properties": {"suggestions": {"type": "array", "items": {"type": "string"}}},
    "required": ["suggestions"],
    "additionalProperties": False,
}

_SUGGEST_PROMPT = """\
ショッピングアシスタントとユーザーの会話の直近のやり取りです。

<user>{user}</user>
<assistant>{ai}</assistant>

ユーザーが次に送りそうな返信の候補を 3 つ作ってください。
- ユーザー本人の言葉として、短く自然な日本語（各 25 文字以内）
- アシスタントの問いかけがあれば、1 つ目はそれへの具体的な答えの例にする
- 3 つは互いに違う方向性にする（例: 条件を伝える / 別の商品と比べる / カートに入れる）
"""


def _message_text(message: dict[str, Any]) -> str:
    """履歴の 1 メッセージから、人が読むテキストだけを取り出す（画面操作メモは除く）。"""
    content = message["content"]
    if isinstance(content, str):
        return content.strip()
    texts = []
    for block in content:
        if _btype(block) != "text":
            continue
        t = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
        if t and not t.startswith("[画面操作メモ]"):
            texts.append(t)
    return "\n".join(texts).strip()


def _shorten(text: str, limit: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _rollback(session: Session, start_len: int, notes: list[str]) -> None:
    """失敗したターンを履歴の末尾から取り除き、画面操作メモは次のターンに持ち越す。"""
    del session.messages[start_len:]
    session.pending_notes = notes + session.pending_notes


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
