"""ターミナルで会話する CLI。

  python -m shopping_ai
"""

from __future__ import annotations

import asyncio
import json
import sys

from .agent import ShoppingAgent
from .session import Session

HELP = "コマンド: /cart カート表示, /prefs 記憶した好み, /reset 会話リセット, /quit 終了"


async def main() -> None:
    agent = ShoppingAgent()
    session = Session()
    print("🛍  おかいものAI です。何をお探しですか？（" + HELP + "）\n")
    while True:
        try:
            line = await asyncio.to_thread(input, "あなた> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        line = line.strip()
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return
        if line == "/cart":
            print(json.dumps(session.cart.to_dict(), ensure_ascii=False, indent=2))
            continue
        if line == "/prefs":
            print(json.dumps(session.preferences, ensure_ascii=False, indent=2))
            continue
        if line == "/reset":
            session = Session()
            print("会話をリセットしました。")
            continue
        if line == "/help":
            print(HELP)
            continue

        print("AI> ", end="", flush=True)
        status_shown = False
        async for ev in agent.chat(session, line):
            if ev["type"] == "text":
                if status_shown:
                    sys.stdout.write("\r\033[K")
                    status_shown = False
                sys.stdout.write(ev["text"])
                sys.stdout.flush()
            elif ev["type"] == "status" and ev["text"]:
                sys.stdout.write(f"\r\033[K\033[2m（{ev['text']}）\033[0m")
                sys.stdout.flush()
                status_shown = True
            elif ev["type"] == "cart":
                c = ev["cart"]
                sys.stdout.write(f"\n\033[2m[カート: {c['item_count']}点 / 合計 ¥{c['total_jpy']:,}]\033[0m\n")
            elif ev["type"] == "error":
                sys.stdout.write(f"\n⚠ {ev['text']}")
        if status_shown:
            sys.stdout.write("\r\033[K")
        print("\n")


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
