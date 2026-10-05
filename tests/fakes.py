"""API キーなしでエージェントのループを検証するための偽 Claude クライアント。"""

from __future__ import annotations

from types import SimpleNamespace as NS
from typing import Any


def text(t: str) -> NS:
    return NS(type="text", text=t)


def tool_use(id_: str, name: str, input_: dict[str, Any]) -> NS:
    return NS(type="tool_use", id=id_, name=name, input=input_)


def message(content: list[NS], stop_reason: str = "end_turn") -> NS:
    return NS(content=content, stop_reason=stop_reason,
              usage=NS(input_tokens=100, output_tokens=20, cache_read_input_tokens=50))


class _Stream:
    def __init__(self, msg: NS):
        self._msg = msg

    async def __aenter__(self) -> "_Stream":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    def __aiter__(self):
        return self._events()

    async def _events(self):
        for block in self._msg.content:
            yield NS(type="content_block_start", content_block=block)
            if block.type == "text":
                yield NS(type="text", text=block.text)
            yield NS(type="content_block_stop", content_block=block)

    async def get_final_message(self) -> NS:
        return self._msg


class FakeClient:
    """scripted の順にレスポンスを返す。送られたリクエストは calls に記録する。"""

    def __init__(self, scripted: list[NS]):
        self._scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **params: Any) -> _Stream:
        # messages はその後も追記されるので、呼び出し時点のコピーを残す
        self.calls.append({**params, "messages": list(params["messages"])})
        return _Stream(self._scripted.pop(0))
