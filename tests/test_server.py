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
