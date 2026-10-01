"""Daily KPI data by segment, with known incidents injected so detection can be evaluated.

Output:
    data/kpis.parquet      date, channel, region, platform, sessions, orders, revenue
    data/incidents.json    ground truth for the injected incidents
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
CHANNELS = ["organic", "paid_search", "social", "email"]
REGIONS = ["NA", "EU", "APAC"]
PLATFORMS = ["web", "ios", "android"]

INCIDENTS = [
    {
        "name": "iOS app release breaks checkout",
        "start": "2026-08-18",
        "days": 3,
        "filter": {"platform": "ios"},
        "effect": {"orders": 0.45, "revenue": 0.45},
        "expected_metric": "conversion_rate",
        "expected_dimension": "platform",
    },
    {
        "name": "Android tracking outage",
        "start": "2026-09-02",
        "days": 1,
        "filter": {"platform": "android"},
        "effect": {"sessions": 0.15, "orders": 0.15, "revenue": 0.15},
        "expected_metric": "sessions",
        "expected_dimension": "platform",
    },
    {
        "name": "Paid search campaign launch",
        "start": "2026-09-15",
        "days": 4,
        "filter": {"channel": "paid_search"},
        "effect": {"sessions": 2.2, "orders": 2.0, "revenue": 1.9},
        "expected_metric": "sessions",
        "expected_dimension": "channel",
    },
]


def generate(seed: int = 21, start: str = "2026-05-01", end: str = "2026-09-30") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    base_sessions = {"organic": 27000, "paid_search": 15000, "social": 10500, "email": 4500}
    region_share = {"NA": 0.5, "EU": 0.32, "APAC": 0.18}
    platform_share = {"web": 0.55, "ios": 0.27, "android": 0.18}
    conv = {"organic": 0.031, "paid_search": 0.026, "social": 0.014, "email": 0.048}
    rows = []
    for d, c, r, p in product(dates, CHANNELS, REGIONS, PLATFORMS):
        t = (d - dates[0]).days
        weekday = 1.0 + 0.12 * (d.weekday() >= 5) * (-1 if c == "email" else 1)
        mean_sessions = (
            base_sessions[c] * region_share[r] * platform_share[p] * weekday * (1 + 0.001 * t)
        )
        sessions = rng.poisson(mean_sessions)
        orders = rng.binomial(sessions, conv[c] * (1.08 if p == "web" else 1.0))
        aov = {"NA": 82, "EU": 74, "APAC": 61}[r]
        revenue = float(np.round(orders * aov * rng.lognormal(0, 0.05), 2))
        rows.append((d, c, r, p, sessions, orders, revenue))
    df = pd.DataFrame(
        rows, columns=["date", "channel", "region", "platform", "sessions", "orders", "revenue"]
    )
    for inc in INCIDENTS:
        window = pd.date_range(inc["start"], periods=inc["days"], freq="D")
        mask = df["date"].isin(window)
        for col, val in inc["filter"].items():
            mask &= df[col] == val
        for metric, factor in inc["effect"].items():
            df.loc[mask, metric] = (df.loc[mask, metric] * factor).round().astype(df[metric].dtype)
    return df


def build() -> None:
    DATA.mkdir(exist_ok=True)
    df = generate()
    df.to_parquet(DATA / "kpis.parquet", index=False)
    (DATA / "incidents.json").write_text(json.dumps(INCIDENTS, indent=2))
    print(f"{len(df):,} rows, {df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d} -> {DATA}/")


def load() -> pd.DataFrame:
    return pd.read_parquet(DATA / "kpis.parquet")


if __name__ == "__main__":
    build()
