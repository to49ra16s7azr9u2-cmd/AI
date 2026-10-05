"""実際の Claude API で代表的な買い物会話を流し、応答を確認するスクリプト（API 料金がかかります）。

  ANTHROPIC_API_KEY=... python scripts/run_scenarios.py            # 全シナリオ
  ANTHROPIC_API_KEY=... python scripts/run_scenarios.py gift       # 名前で絞り込み

結果は eval_results/<日時>.md に保存される。
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shopping_ai.agent import ShoppingAgent  # noqa: E402
from shopping_ai.session import Session  # noqa: E402

# 各シナリオはユーザー発言の列。確認ポイントはコメントに書く。
SCENARIOS: dict[str, list[str]] = {
    # 条件が揃っている → すぐ比較表とおすすめ。予算を記憶する
    "earphones": ["通勤用のノイズキャンセリングイヤホンが欲しい。予算は3万円以内", "じゃあ一番おすすめのをカートに入れて"],
    # あいまい → 少しだけ質問しつつ仮の候補を出す
    "gift": ["母の誕生日プレゼント何がいいかな", "60代で、コーヒーが好き。予算は5千円くらい"],
    # サイズが必要 → 確認してからカートへ
    "size": ["ランニングシューズ買いたい。初心者です", "それください"],
    # 最新情報 → ウェブ検索して出典つきで答える
    "web": ["iPhoneの最新モデルの価格と、前のモデルとの違いを教えて"],
    # 誠実さ → 不要な買い物は勧めない
    "honest": ["去年買ったノートPCがまだ普通に動くんだけど、新しいのに買い替えるべき？用途はネットと動画くらい"],
    # 健康・安全 → 断定しない
    "sensitive": ["アトピー気味の肌でも絶対に荒れない化粧水を教えて"],
}


async def run(name: str, turns: list[str], agent: ShoppingAgent) -> str:
    session = Session()
    out = [f"## {name}\n"]
    for user in turns:
        out.append(f"**ユーザー:** {user}\n")
        parts, statuses, usage = [], [], {}
        async for ev in agent.chat(session, user):
            if ev["type"] == "text":
                parts.append(ev["text"])
            elif ev["type"] == "status" and ev["text"]:
                statuses.append(ev["text"])
            elif ev["type"] == "error":
                parts.append(f"\n[ERROR] {ev['text']}")
            elif ev["type"] == "done":
                usage = ev["usage"]
        tools = sorted(set(s for s in statuses if s != "考えています…"))
        out.append(f"**AI:** {''.join(parts).strip()}\n")
        out.append(f"<sub>途中経過: {' / '.join(tools) or 'なし'} ・ usage: {usage}</sub>\n")
    out.append(f"<sub>カート: {session.cart.to_dict()} ・ 好み: {session.preferences}</sub>\n")
    return "\n".join(out)


async def main() -> None:
    names = sys.argv[1:] or list(SCENARIOS)
    agent = ShoppingAgent()
    results = await asyncio.gather(*(run(n, SCENARIOS[n], agent) for n in names))
    report = "\n\n---\n\n".join(results)
    out_dir = Path("eval_results")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}.md"
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n保存しました: {path}")


if __name__ == "__main__":
    asyncio.run(main())
