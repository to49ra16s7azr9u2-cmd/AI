"""実際の Claude API で代表的な買い物会話を流し、AI 採点で品質を測るスクリプト（API 料金がかかります）。

  ANTHROPIC_API_KEY=... python scripts/run_scenarios.py              # 全シナリオ + 採点
  ANTHROPIC_API_KEY=... python scripts/run_scenarios.py gift web     # 名前で絞り込み
  ANTHROPIC_API_KEY=... python scripts/run_scenarios.py --no-judge   # 会話だけ（採点なし）

結果は eval_results/<日時>.md に保存される。プロンプトや設定を変えたら再実行して、
シナリオごとの点数を前回と比べる（AI の応答には揺らぎがあるので、1〜2 点の差は誤差のこともある）。
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic  # noqa: E402

from shopping_ai.agent import ShoppingAgent  # noqa: E402
from shopping_ai.session import Session  # noqa: E402


@dataclass
class Scenario:
    turns: list[str]
    # このシナリオで特に見るポイント（採点基準）
    rubric: list[str]


# すべてのシナリオに共通する採点基準
COMMON_RUBRIC = [
    "結論が冒頭にあり、長さが質問に見合っている（前置き・定型の締めの文句がない）",
    "価格・在庫・スペックをツール結果や出典なしに断定していない（でっち上げがない）",
    "自然で親しみやすい日本語で、ベテラン店員のように相手の状況を汲んでいる",
]

SCENARIOS: dict[str, Scenario] = {
    "earphones": Scenario(
        ["通勤用のノイズキャンセリングイヤホンが欲しい。予算は3万円以内", "じゃあ一番おすすめのをカートに入れて"],
        [
            "予算内で、ノイズキャンセリング搭載の商品だけを勧めている（非搭載モデルを本命にしていない）",
            "比較に表を使い、弱点や注意点も書いている",
            "予算を remember_preference で記憶し、2 ターン目で実際にカートへ追加している",
        ],
    ),
    "gift": Scenario(
        ["母の誕生日プレゼント何がいいかな", "60代で、コーヒーが好き。予算は5千円くらい"],
        [
            "1 ターン目は質問を 1〜3 個に絞り、質問だけで終わらず方向性や仮の候補も示している",
            "2 ターン目は 60 代・コーヒー好き・5 千円の条件に合う商品を、理由つきで 2〜4 点に絞っている",
            "ギフトとしての配慮（熨斗・ラッピング・日持ち・配送日数など）に触れている",
        ],
    ),
    "size": Scenario(
        ["ランニングシューズ買いたい。初心者です", "それください"],
        [
            "初心者向けの選び方のポイント（クッション性・サイズ感など）を簡潔に添えている",
            "サイズ・色が未確定のままカートに追加せず、確認している",
        ],
    ),
    "web": Scenario(
        ["iPhoneの最新モデルの価格と、前のモデルとの違いを教えて"],
        [
            "ウェブ検索で調べ、出典（サイト名やリンク）を示している",
            "価格は時点つき・変動しうると明記している",
            "当ストアで扱っていない商品だと分かる書き方で、購入はリンクで案内している",
        ],
    ),
    "honest": Scenario(
        ["去年買ったノートPCがまだ普通に動くんだけど、新しいのに買い替えるべき？用途はネットと動画くらい"],
        [
            "用途を踏まえて、今は買い替え不要であることを率直に伝えている（売りつけていない）",
            "買い替えを検討すべきサインや、延命の工夫など実用的な助言がある",
        ],
    ),
    "sensitive": Scenario(
        ["アトピー気味の肌でも絶対に荒れない化粧水を教えて"],
        [
            "「絶対に荒れない」とは断定せず、その理由を穏やかに伝えている",
            "敏感肌向けの候補を挙げつつ、パッチテストや皮膚科への相談などの注意点を添えている",
        ],
    ),
    "vague_then_change": Scenario(
        ["一人暮らし始めるんだけど、家電で最初に買うべきものある？", "掃除が苦手。予算は全部で5万円"],
        [
            "一人暮らしで優先度の高いものから整理して提案している",
            "2 ターン目で条件（掃除が苦手・合計 5 万円）を反映し、合計金額が予算に収まる組み合わせを示している",
        ],
    ),
}


# ---------------------------------------------------------------------------
# 会話の実行
# ---------------------------------------------------------------------------


async def run_conversation(scenario: Scenario, agent: ShoppingAgent) -> tuple[str, Session]:
    session = Session()
    lines: list[str] = []
    for user in scenario.turns:
        lines.append(f"**ユーザー:** {user}\n")
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
        tool_log = list(dict.fromkeys(s for s in statuses if s != "考えています…"))
        lines.append(f"**AI:** {''.join(parts).strip()}\n")
        lines.append(f"<sub>途中経過: {' / '.join(tool_log) or 'なし'} ・ usage: {usage}</sub>\n")
    lines.append(f"<sub>最終カート: {json.dumps(session.cart.to_dict(), ensure_ascii=False)} ・ "
                 f"記憶した好み: {json.dumps(session.preferences, ensure_ascii=False)}</sub>\n")
    return "\n".join(lines), session


# ---------------------------------------------------------------------------
# AI による採点
# ---------------------------------------------------------------------------

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion": {"type": "string"},
                    "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "reason": {"type": "string"},
                },
                "required": ["criterion", "score", "reason"],
                "additionalProperties": False,
            },
        },
        "biggest_issue": {"type": "string"},
    },
    "required": ["scores", "biggest_issue"],
    "additionalProperties": False,
}

JUDGE_PROMPT = """\
あなたはオンラインストアの接客品質を審査する厳しい審査員です。
以下はショッピング AI アシスタントとユーザーの会話記録です（途中経過はツール使用のログ、最後の行は最終的なカートと記憶した好み）。

<transcript>
{transcript}
</transcript>

次の各基準について 1〜5 点で採点してください。
5 = 一流の店員でもこれ以上は難しい / 4 = 良いが小さな改善点あり / 3 = 及第点 / 2 = 明確な問題あり / 1 = 基準を満たしていない

<criteria>
{criteria}
</criteria>

各基準について、根拠を会話から具体的に挙げてから点数を決めること。甘く付けないこと。
最後に、この会話で最も改善すべき点を 1 つ biggest_issue に書くこと。
"""


async def judge(client: anthropic.AsyncAnthropic, model: str, transcript: str, criteria: list[str]) -> dict:
    resp = await client.messages.create(
        model=model,
        max_tokens=16000,
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                transcript=transcript, criteria="\n".join(f"{i + 1}. {c}" for i, c in enumerate(criteria))
            ),
        }],
    )
    if resp.stop_reason != "end_turn":
        raise RuntimeError(f"judge stopped with {resp.stop_reason}")
    return json.loads(next(b.text for b in resp.content if b.type == "text"))


# ---------------------------------------------------------------------------


async def run_one(name: str, scenario: Scenario, agent: ShoppingAgent, do_judge: bool) -> tuple[str, float | None]:
    transcript, _ = await run_conversation(scenario, agent)
    section = [f"## {name}\n", transcript]
    avg = None
    if do_judge:
        try:
            result = await judge(agent.client, agent.settings.model, transcript, scenario.rubric + COMMON_RUBRIC)
            scores = result["scores"]
            avg = sum(s["score"] for s in scores) / len(scores)
            section.append(f"\n**採点: {avg:.2f} / 5**\n")
            section.append("| 基準 | 点 | 理由 |\n|---|---|---|")
            for s in scores:
                section.append(f"| {s['criterion']} | {s['score']} | {s['reason']} |")
            section.append(f"\n**最大の改善点:** {result['biggest_issue']}\n")
        except Exception as e:  # 採点の失敗で全体を止めない
            section.append(f"\n[採点エラー] {e}\n")
    return "\n".join(section), avg


async def main() -> None:
    args = sys.argv[1:]
    do_judge = "--no-judge" not in args
    names = [a for a in args if not a.startswith("--")] or list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        sys.exit(f"不明なシナリオ: {unknown}（選択肢: {list(SCENARIOS)}）")

    agent = ShoppingAgent()
    results = await asyncio.gather(*(run_one(n, SCENARIOS[n], agent, do_judge) for n in names))

    summary = ["# おかいものAI 品質チェック", "",
               f"- 日時: {datetime.now():%Y-%m-%d %H:%M}", f"- モデル: {agent.settings.model} / effort: {agent.settings.effort}", ""]
    scored = [(n, a) for n, (_, a) in zip(names, results) if a is not None]
    if scored:
        summary += ["| シナリオ | 点数 (5点満点) |", "|---|---|"]
        summary += [f"| {n} | {a:.2f} |" for n, a in scored]
        summary.append(f"| **平均** | **{sum(a for _, a in scored) / len(scored):.2f}** |")
    report = "\n".join(summary) + "\n\n---\n\n" + "\n\n---\n\n".join(r for r, _ in results)

    out_dir = Path("eval_results")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}.md"
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n保存しました: {path}")


if __name__ == "__main__":
    asyncio.run(main())
