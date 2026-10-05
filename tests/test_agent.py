import asyncio

from shopping_ai.agent import ShoppingAgent, sanitize_for_history
from shopping_ai.config import Settings
from shopping_ai.session import Session

from .fakes import FakeClient, NS, message, text, tool_use


def run(agent, session, msg, images=None):
    async def go():
        return [ev async for ev in agent.chat(session, msg, images)]

    return asyncio.run(go())


def make_agent(scripted, **settings):
    client = FakeClient(scripted)
    return ShoppingAgent(Settings(**settings), client=client), client


def test_simple_reply_streams_text_and_builds_history():
    agent, client = make_agent([message([text("こんにちは！何をお探しですか？")])])
    s = Session()
    events = run(agent, s, "こんにちは")
    assert "".join(e["text"] for e in events if e["type"] == "text") == "こんにちは！何をお探しですか？"
    assert events[-1]["type"] == "done"
    assert [m["role"] for m in s.messages] == ["user", "assistant"]


def test_request_params():
    agent, client = make_agent([message([text("ok")])])
    run(agent, Session(), "hi")
    p = client.calls[0]
    assert p["model"] == "claude-opus-5-5"
    assert p["thinking"] == {"type": "adaptive"}
    assert p["output_config"] == {"effort": "high"}
    assert p["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in p["betas"]
    assert "tool_choice" not in p
    names = [t.get("name") for t in p["tools"]]
    assert "web_search" in names and "search_catalog" in names


def test_web_can_be_disabled():
    agent, client = make_agent([message([text("ok")])], enable_web=False, enable_compaction=False)
    run(agent, Session(), "hi")
    p = client.calls[0]
    assert "web_search" not in [t.get("name") for t in p["tools"]]
    assert "context_management" not in p and p["betas"] == ["server-side-fallback-2026-07-01"]


def test_tool_loop_adds_to_cart():
    agent, client = make_agent([
        message([text("お探ししますね。"), tool_use("t1", "search_catalog", {"query": "ケトル"})], "tool_use"),
        message([tool_use("t2", "add_to_cart", {"product_id": "EL-007", "quantity": 1})], "tool_use"),
        message([text("カートに入れました！")]),
    ])
    s = Session()
    events = run(agent, s, "ケトルをカートに入れて")
    assert any(e["type"] == "cart" and e["cart"]["total_jpy"] == 6980 for e in events)
    assert s.cart.total == 6980
    # 2 回目の呼び出しには t1 の結果が含まれている
    second = client.calls[1]["messages"]
    assert second[-1]["role"] == "user"
    assert second[-1]["content"][0]["tool_use_id"] == "t1"
    assert "EL-007" in second[-1]["content"][0]["content"]
    assert [m["role"] for m in s.messages] == ["user", "assistant", "user", "assistant", "user", "assistant"]


def test_tool_error_is_reported_to_model():
    agent, client = make_agent([
        message([tool_use("t1", "add_to_cart", {"product_id": "FA-001", "quantity": 1})], "tool_use"),
        message([text("サイズを教えてください")]),
    ])
    s = Session()
    run(agent, s, "ダウンください")
    result = client.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    assert s.cart.items == []


def test_pause_turn_continues_without_user_message():
    agent, client = make_agent([
        message([text("調べています")], "pause_turn"),
        message([text("結果です")]),
    ])
    s = Session()
    run(agent, s, "最新の相場は？")
    assert len(client.calls) == 2
    assert client.calls[1]["messages"][-1]["role"] == "assistant"


def test_refusal_rolls_back_turn():
    agent, _ = make_agent([message([], "refusal")])
    s = Session()
    events = run(agent, s, "???")
    assert any(e["type"] == "error" for e in events)
    assert s.messages == []


def test_api_error_rolls_back_and_reports():
    import anthropic
    import httpx2

    class Boom(FakeClient):
        def _stream(self, **params):
            req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
            raise anthropic.APIConnectionError(request=req)

    agent = ShoppingAgent(Settings(), client=Boom([]))
    s = Session()
    events = run(agent, s, "hi")
    assert events[-2]["type"] == "error" and "接続" in events[-2]["text"]
    assert s.messages == []


def test_images_become_image_blocks():
    agent, client = make_agent([message([text("素敵なバッグですね")])])
    run(agent, Session(), "これに似たもの", [{"media_type": "image/png", "data": "aGVsbG8="}])
    content = client.calls[0]["messages"][0]["content"]
    assert content[0]["type"] == "image" and content[-1]["type"] == "text"


def test_sanitize_after_midstream_fallback():
    content = [
        NS(type="thinking", thinking=""),
        NS(type="text", text="途中まで"),
        NS(type="server_tool_use", id="s1", name="web_search", input={}),
        NS(type="tool_use", id="u1", name="view_cart", input={}),
        NS(type="fallback"),
        NS(type="thinking", thinking=""),
        NS(type="text", text="続き"),
    ]
    kept = [b.type for b in sanitize_for_history(content)]
    assert kept == ["text", "fallback", "thinking", "text"]
    # フォールバックが無ければ何も変えない
    plain = [NS(type="thinking"), NS(type="text")]
    assert sanitize_for_history(plain) == plain
