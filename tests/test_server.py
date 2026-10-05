import json

from fastapi.testclient import TestClient

from shopping_ai.agent import ShoppingAgent
from shopping_ai.config import Settings
from shopping_ai.server import create_app

from .fakes import FakeClient, message, text, tool_use


def make_client(scripted):
    agent = ShoppingAgent(Settings(), client=FakeClient(scripted))
    return TestClient(create_app(agent=agent))


def parse_sse(body: str):
    return [json.loads(line[6:]) for line in body.split("\n\n") if line.startswith("data: ")]


def test_index_and_health():
    c = make_client([])
    assert c.get("/").status_code == 200
    assert c.get("/api/health").json()["ok"] is True


def test_chat_stream_and_session_state():
    c = make_client([
        message([tool_use("t1", "add_to_cart", {"product_id": "EL-010", "quantity": 2})], "tool_use"),
        message([text("追加しました")]),
    ])
    r = c.post("/api/chat", json={"message": "モバイルバッテリー2個ください"})
    assert r.status_code == 200
    events = parse_sse(r.text)
    assert events[0]["type"] == "session"
    sid = events[0]["session_id"]
    assert any(e["type"] == "cart" for e in events)
    assert events[-1]["type"] == "done"
    assert events[-2]["type"] == "suggestions" and len(events[-2]["items"]) == 3
    state = c.get(f"/api/session/{sid}").json()
    assert state["cart"]["total_jpy"] == 6960
    state = c.delete(f"/api/session/{sid}/cart/EL-010").json()
    assert state["cart"]["items"] == []


def test_rejects_bad_input():
    c = make_client([])
    assert c.post("/api/chat", json={"message": "  "}).status_code == 400
    bad_img = {"message": "x", "images": [{"media_type": "image/bmp", "data": "aGVsbG8="}]}
    assert c.post("/api/chat", json=bad_img).status_code == 400
    bad_b64 = {"message": "x", "images": [{"media_type": "image/png", "data": "!!!"}]}
    assert c.post("/api/chat", json=bad_b64).status_code == 400


def test_products_endpoint_and_ui_cart_notes():
    c = make_client([message([text("いいですね")])])
    r = c.get("/api/products", params={"ids": "EL-001,FA-004,NOPE,EL-001"}).json()
    assert [p["id"] for p in r["products"]] == ["EL-001", "FA-004"]
    assert "sizes" in r["products"][1]

    # サイズ必須の商品はオプションなしだと 400
    assert c.post("/api/session/s1/cart", json={"product_id": "FA-004"}).status_code == 400
    state = c.post("/api/session/s1/cart", json={"product_id": "FA-004", "option": "26.5 / ホワイト"}).json()
    assert state["cart"]["total_jpy"] == 14300

    # 画面操作は次の発言と一緒にモデルへ伝わる
    c.post("/api/chat", json={"session_id": "s1", "message": "これで大丈夫？"})
    sent = c.app.state.agent.client.calls[0]["messages"][0]["content"][0]["text"]
    assert sent.startswith("[画面操作メモ]") and "FA-004" in sent
