"""Metric definitions, anomaly detection and segment drill-down (deterministic, no LLM).

Anomalies are judged against the same weekday over the previous 6 weeks using a robust z-score
(median / MAD), which ignores weekly seasonality and is not distorted by earlier incidents.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

ADDITIVE = ["sessions", "orders", "revenue"]
RATIOS = {"conversion_rate": ("orders", "sessions"), "aov": ("revenue", "orders")}
METRICS = ADDITIVE + list(RATIOS)
DIMENSIONS = ["channel", "region", "platform"]
Z_THRESHOLD = 4.5
BASELINE_WEEKS = 6


def daily(df: pd.DataFrame, filters: dict | None = None) -> pd.DataFrame:
    """Daily totals (optionally filtered) with ratio metrics derived from the sums."""
    for col, val in (filters or {}).items():
        df = df[df[col] == val]
    out = df.groupby("date")[ADDITIVE].sum()
    for name, (num, den) in RATIOS.items():
        out[name] = out[num] / out[den].replace(0, np.nan)
    return out


def baseline_dates(date: pd.Timestamp) -> list[pd.Timestamp]:
    return [date - pd.Timedelta(weeks=w) for w in range(1, BASELINE_WEEKS + 1)]


@dataclass
class Anomaly:
    date: str
    metric: str
    value: float
    expected: float
    pct_change: float
    z: float

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def detect(df: pd.DataFrame, since: str | None = None) -> list[Anomaly]:
    d = daily(df)
    found = []
    for date in d.index[BASELINE_WEEKS * 7 :]:
        if since and date < pd.Timestamp(since):
            continue
        base = d.loc[baseline_dates(date)]
        for m in METRICS:
            med = base[m].median()
            mad = (base[m] - med).abs().median() * 1.4826
            scale = max(mad, abs(med) * 0.02)  # floor: ignore sub-2% wiggles on very stable series
            z = (d.at[date, m] - med) / scale
            if abs(z) >= Z_THRESHOLD:
                found.append(
                    Anomaly(
                        f"{date:%Y-%m-%d}",
                        m,
                        float(d.at[date, m]),
                        float(med),
                        float(d.at[date, m] / med - 1),
                        float(z),
                    )
                )
    return found


def group_incidents(anomalies: list[Anomaly]) -> list[list[Anomaly]]:
    """Merge anomalies at most two days apart into one incident."""
    if not anomalies:
        return []
    by_date = sorted({a.date for a in anomalies})
    groups, current = [], [by_date[0]]
    for prev, nxt in zip(by_date, by_date[1:], strict=False):
        if (pd.Timestamp(nxt) - pd.Timestamp(prev)).days <= 2:
            current.append(nxt)
        else:
            groups.append(current)
            current = [nxt]
    groups.append(current)
    return [[a for a in anomalies if a.date in g] for g in groups]


def breakdown(df: pd.DataFrame, metric: str, date: str, dimension: str) -> pd.DataFrame:
    """How much each segment of ``dimension`` contributed to the metric's change vs. baseline.

    Additive metrics: contribution = segment change / total change.
    Ratio metrics: rate-change effect = (segment rate − baseline rate) × segment share of the
    denominator, which isolates segments whose behaviour changed from pure mix shifts.
    """
    day = pd.Timestamp(date)
    cur = df[df["date"] == day].groupby(dimension)[ADDITIVE].sum()
    base = (
        df[df["date"].isin(baseline_dates(day))]
        .groupby([dimension, "date"])[ADDITIVE]
        .sum()
        .groupby(level=0)
        .median()
    )
    out = pd.DataFrame(index=cur.index)
    if metric in ADDITIVE:
        out["current"] = cur[metric]
        out["baseline"] = base[metric]
        out["change"] = out["current"] - out["baseline"]
        total = out["change"].sum()
        out["share_of_change"] = out["change"] / total if total else 0.0
    else:
        num, den = RATIOS[metric]
        out["current"] = cur[num] / cur[den]
        out["baseline"] = base[num] / base[den]
        weight = cur[den] / cur[den].sum()
        out["change"] = (out["current"] - out["baseline"]) * weight
        total = out["change"].sum()
        out["share_of_change"] = out["change"] / total if total else 0.0
    out["pct_change"] = out["current"] / out["baseline"] - 1
    return out.sort_values("share_of_change", ascending=False).round(4)


def top_driver(df: pd.DataFrame, metric: str, date: str) -> tuple[str, str, float]:
    """The single (dimension, segment) that explains the largest share of the change."""
    best = ("", "", 0.0)
    for dim in DIMENSIONS:
        b = breakdown(df, metric, date, dim)
        seg = b["share_of_change"].idxmax()
        share = float(b.loc[seg, "share_of_change"])
        if share > best[2]:
            best = (dim, seg, share)
    return best
