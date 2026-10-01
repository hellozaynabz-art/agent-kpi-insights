"""Investigation agent: detection is deterministic; Claude investigates causes and writes the brief.

Splitting it this way keeps the numbers trustworthy (code finds anomalies and computes every
figure) while the model does what it is good at: forming and testing hypotheses across
segments, and explaining the findings for executives.
"""

from __future__ import annotations

import json

import pandas as pd

from kpi_agent import metrics
from kpi_agent.llm import Claude, text_of

SYSTEM = """You are an analytics lead investigating KPI anomalies for an e-commerce business.

You get a list of detected incidents. For each one:
1. Use get_breakdown on the most anomalous metric and day across every dimension (channel,
   region, platform) to find which segment explains the change.
2. Check related metrics to tell apart causes: a tracking problem moves sessions and orders
   together; a checkout problem drops orders and conversion while sessions hold; a campaign
   adds sessions in one channel.
3. Use get_trend to check when it started and whether it has recovered.

Then write the brief in Markdown:
- `# KPI Brief`, a one-paragraph executive summary first
- One `##` section per incident: what happened (with figures), most likely cause and the evidence,
  business impact (lost or gained orders and revenue), and a recommended next step with an owner team
- Only state figures that came from tool results. Say how confident you are in each cause.
Keep it under 450 words."""

TOOLS = [
    {
        "name": "get_breakdown",
        "description": "Split a metric's change on one day (vs. the same weekday in prior weeks) "
        "by the segments of a dimension. share_of_change shows which segments explain the move.",
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "enum": metrics.METRICS},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "dimension": {"type": "string", "enum": metrics.DIMENSIONS},
            },
            "required": ["metric", "date", "dimension"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "get_trend",
        "description": "Daily values of a metric between two dates, optionally for one segment "
        "(e.g. dimension='platform', segment='ios').",
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "enum": metrics.METRICS},
                "start": {"type": "string", "description": "YYYY-MM-DD"},
                "end": {"type": "string", "description": "YYYY-MM-DD"},
                "dimension": {"type": "string", "enum": [*metrics.DIMENSIONS, "none"]},
                "segment": {"type": "string", "description": "Segment value, or 'all'"},
            },
            "required": ["metric", "start", "end", "dimension", "segment"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def incidents_payload(groups: list[list[metrics.Anomaly]]) -> str:
    return json.dumps(
        [{"incident": i + 1, "anomalies": [a.to_dict() for a in g]} for i, g in enumerate(groups)],
        indent=1,
    )


def run_tool(df: pd.DataFrame, name: str, args: dict) -> str:
    if name == "get_breakdown":
        b = metrics.breakdown(df, args["metric"], args["date"], args["dimension"])
        return b.to_markdown()
    if name == "get_trend":
        filters = {} if args["dimension"] == "none" else {args["dimension"]: args["segment"]}
        d = metrics.daily(df, filters).loc[args["start"] : args["end"], [args["metric"]]]
        return d.round(4).to_markdown()
    raise ValueError(f"unknown tool {name}")


def investigate(
    df: pd.DataFrame,
    groups: list[list[metrics.Anomaly]],
    claude: Claude | None = None,
    max_turns: int = 25,
    on_tool=None,
) -> str:
    claude = claude or Claude()
    messages = [
        {
            "role": "user",
            "content": f"Detected incidents (expected = median of the same weekday over the prior "
            f"{metrics.BASELINE_WEEKS} weeks):\n{incidents_payload(groups)}\n\nInvestigate and write the brief.",
        }
    ]
    for _ in range(max_turns):
        response = claude.create(system=SYSTEM, tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return text_of(response)
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if on_tool:
                on_tool(block.name, block.input)
            try:
                out, err = run_tool(df, block.name, block.input), False
            except (KeyError, ValueError) as e:
                out, err = f"Error: {e}", True
            results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": out, "is_error": err}
            )
        messages.append({"role": "user", "content": results})
    return "Investigation stopped at the step limit."


def _fmt(metric: str, value: float) -> str:
    if metric == "conversion_rate":
        return f"{value:.2%}"
    if metric in ("revenue", "aov"):
        return f"${value:,.0f}"
    return f"{value:,.0f}"


def rule_based_brief(df: pd.DataFrame, groups: list[list[metrics.Anomaly]]) -> str:
    """No-LLM fallback: same detection and drill-down, templated wording."""
    lines = ["# KPI Brief (rule-based)", "", f"{len(groups)} incident(s) detected.", ""]
    for i, g in enumerate(groups, 1):
        worst = max(g, key=lambda a: abs(a.z))
        dim, seg, share = metrics.top_driver(df, worst.metric, worst.date)
        moved = ", ".join(sorted({a.metric for a in g}))
        lines += [
            f"## Incident {i}: {g[0].date} to {g[-1].date}",
            f"- Metrics affected: {moved}",
            f"- Largest move: **{worst.metric}** {worst.pct_change:+.1%} on {worst.date} "
            f"({_fmt(worst.metric, worst.value)} vs. {_fmt(worst.metric, worst.expected)} expected)",
            # Shares above 100% mean other segments moved the opposite way.
            f"- Top driver: **{dim} = {seg}** explains {min(share, 1):.0%} of the change",
            "",
        ]
    return "\n".join(lines)
