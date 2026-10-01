"""Daily run: detect anomalies, investigate, write ``reports/brief_<date>.md``.

uv run kpi-agent                 # Claude investigates and writes the brief
uv run kpi-agent --no-llm        # rule-based brief, no API key needed
uv run kpi-agent --since 2026-09-01
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from kpi_agent import agent, data, metrics
from kpi_agent.llm import Claude

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", help="only report anomalies on or after this date (YYYY-MM-DD)")
    parser.add_argument("--no-llm", action="store_true", help="rule-based brief without the API")
    args = parser.parse_args()

    if not (data.DATA / "kpis.parquet").exists():
        data.build()
    df = data.load()
    groups = metrics.group_incidents(metrics.detect(df, since=args.since))
    console.print(f"[bold]{len(groups)} incident(s) detected[/]")

    if not groups:
        brief = "# KPI Brief\n\nNo anomalies detected. All KPIs are within their normal range."
    elif args.no_llm:
        brief = agent.rule_based_brief(df, groups)
    else:
        claude = Claude()
        with console.status("Investigating…"):
            brief = agent.investigate(
                df,
                groups,
                claude,
                on_tool=lambda n, a: console.print(
                    f"[dim]→ {n}({', '.join(map(str, a.values()))})[/]"
                ),
            )
        console.print(f"[dim]{claude.usage.calls} model calls · ${claude.usage.cost_usd:.3f}[/]")

    out = Path("reports") / f"brief_{date.today():%Y-%m-%d}{'_rules' if args.no_llm else ''}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(brief)
    console.print(Markdown(brief))
    console.print(f"[green]saved {out}[/]")


if __name__ == "__main__":
    main()
