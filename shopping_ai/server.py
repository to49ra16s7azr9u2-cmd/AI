"""Web サーバー（FastAPI）。ブラウザのチャット UI と、ストリーミング（SSE）の API を提供する。

起動: uvicorn shopping_ai.server:app --reload
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .agent import ShoppingAgent
from .config import PACKAGE_DIR
from .session import SessionStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGES = 4
MAX_MESSAGE_CHARS = 8000


class ImageIn(BaseModel):
    media_type: str
    data: str  # base64（data: URL の接頭辞なし）


class ChatIn(BaseModel):
    session_id: str | None = Field(default=None, max_length=64)
    message: str = Field(default="", max_length=MAX_MESSAGE_CHARS)
    images: list[ImageIn] = Field(default_factory=list)


def create_app(agent: ShoppingAgent | None = None, store: SessionStore | None = None) -> FastAPI:
    app = FastAPI(title="おかいものAI")
    app.state.agent = agent or ShoppingAgent()
    app.state.store = store or SessionStore()
    static_dir = PACKAGE_DIR / "static"

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "model": app.state.agent.settings.model, "products": len(app.state.agent.catalog)}

    @app.get("/api/session/{session_id}")
    async def get_session(session_id: str) -> dict[str, Any]:
        session = app.state.store.get(session_id)
        if session is None:
            raise HTTPException(404, "session not found")
        return session.snapshot()

    @app.post("/api/session/{session_id}/reset")
    async def reset_session(session_id: str) -> dict[str, Any]:
        return app.state.store.reset(session_id).snapshot()

    @app.delete("/api/session/{session_id}/cart/{product_id}")
    async def remove_cart_item(session_id: str, product_id: str) -> dict[str, Any]:
        session = app.state.store.get(session_id)
        if session is None:
            raise HTTPException(404, "session not found")
        session.cart.remove(product_id)
        return session.snapshot()

    @app.post("/api/chat")
    async def chat(body: ChatIn) -> StreamingResponse:
        if not body.message.strip() and not body.images:
            raise HTTPException(400, "message is empty")
        if len(body.images) > MAX_IMAGES:
            raise HTTPException(400, f"images: up to {MAX_IMAGES}")
        images = []
        for img in body.images:
            if img.media_type not in ALLOWED_IMAGE_TYPES:
                raise HTTPException(400, f"unsupported image type: {img.media_type}")
            try:
                raw = base64.b64decode(img.data, validate=True)
            except (binascii.Error, ValueError):
                raise HTTPException(400, "invalid base64 image") from None
            if len(raw) > MAX_IMAGE_BYTES:
                raise HTTPException(400, "image too large (max 5MB)")
            images.append({"media_type": img.media_type, "data": img.data})

        session = app.state.store.get_or_create(body.session_id)
        if session.lock.locked():
            raise HTTPException(409, "前の応答を生成中です")

        async def events() -> AsyncIterator[str]:
            async with session.lock:
                yield _sse({"type": "session", **session.snapshot()})
                async for ev in app.state.agent.chat(session, body.message, images):
                    yield _sse(ev)

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    return app


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


app = create_app()
