from types import SimpleNamespace

import pandas as pd
import pytest

from kpi_agent import agent, data, metrics
from kpi_agent.llm import Claude


@pytest.fixture(scope="module")
def df():
    return data.generate()


@pytest.fixture(scope="module")
def groups(df):
    return metrics.group_incidents(metrics.detect(df))


def test_detects_every_injected_incident_without_false_alarms(groups):
    found = [(pd.Timestamp(g[0].date), pd.Timestamp(g[-1].date)) for g in groups]
    for inc in data.INCIDENTS:
        start = pd.Timestamp(inc["start"])
        end = start + pd.Timedelta(days=inc["days"] - 1)
        assert any(s <= end and e >= start for s, e in found), inc["name"]
    assert len(groups) == len(data.INCIDENTS)


def test_drill_down_finds_the_right_segment(df, groups):
    for inc, g in zip(data.INCIDENTS, groups, strict=True):
        worst = max(g, key=lambda a: abs(a.z))
        dim, seg, _ = metrics.top_driver(df, worst.metric, worst.date)
        assert (dim, seg) in inc["filter"].items(), inc["name"]


def test_breakdown_shares_sum_to_one(df):
    b = metrics.breakdown(df, "sessions", "2026-09-15", "channel")
    assert b["share_of_change"].sum() == pytest.approx(1, abs=1e-3)
    assert b.index[0] == "paid_search"


def test_rule_based_brief_mentions_drivers(df, groups):
    brief = agent.rule_based_brief(df, groups)
    assert "paid_search" in brief and "ios" in brief and "android" in brief


def _resp(stop, *blocks):
    return SimpleNamespace(
        stop_reason=stop,
        content=list(blocks),
        usage=SimpleNamespace(input_tokens=1, output_tokens=1, cache_read_input_tokens=0),
    )


def test_agent_loop_calls_tools_and_returns_brief(df, groups):
    scripted = [
        _resp(
            "tool_use",
            SimpleNamespace(
                type="tool_use",
                id="a",
                name="get_breakdown",
                input={"metric": "sessions", "date": "2026-09-15", "dimension": "channel"},
            ),
        ),
        _resp("end_turn", SimpleNamespace(type="text", text="# KPI Brief\nPaid search campaign.")),
    ]
    seen = []

    def create(**kw):
        seen.append(list(kw["messages"]))
        return scripted.pop(0)

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    brief = agent.investigate(df, groups, Claude(client=client))
    assert brief.startswith("# KPI Brief")
    tool_result = seen[1][-1]["content"][0]
    assert "paid_search" in tool_result["content"] and not tool_result["is_error"]
