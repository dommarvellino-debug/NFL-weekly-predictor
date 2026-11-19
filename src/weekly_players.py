"""Current-season weekly player form from nflreadpy.

The Kaggle player features in :mod:`src.player_stats` describe the previous
season. This module adds in-season form: for each player we take the rolling sum
of his *prior* games, then pick each team's leading QB/RB/receiver for every
week. Because the rolling sum shifts by one game, a player's stats from the
current game are never included.

Only offensive positions are used, and all derived values are available before
kickoff. Week 1 has no prior games, so its values are missing (the prior-season
Kaggle features still cover that case).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import pandas as pd

from src import data as nfl_data

WEEKLY_PLAYER_COLUMNS = (
    "qb_ypa",
    "qb_epa_att",
    "rb_ypc",
    "wr_ypr",
    "top3_rec_yards",
)
WEEKLY_PLAYER_NAME_COLUMNS = ("qb_name", "rb_name", "wr_name")

_BASE_COLUMNS = (
    "attempts",
    "completions",
    "passing_yards",
    "passing_epa",
    "carries",
    "rushing_yards",
    "receptions",
    "targets",
    "receiving_yards",
)
_REQUIRED_COLUMNS = (
    "season", "week", "season_type", "game_id", "team",
    "player_id", "player_name", "position", *_BASE_COLUMNS,
)
_OUTPUT_COLUMNS = (
    "season", "week", "team",
    *WEEKLY_PLAYER_NAME_COLUMNS,
    *WEEKLY_PLAYER_COLUMNS,
)


def _normalize_seasons(seasons: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(seasons, int):
        return (seasons,)
    return tuple(sorted({int(season) for season in seasons}))


def _ratio(frame: pd.DataFrame, numerator: str, denominator: str) -> pd.Series:
    """Divide rolling sums, returning NaN when the denominator has no attempts."""
    den = pd.to_numeric(frame[denominator], errors="coerce")
    num = pd.to_numeric(frame[numerator], errors="coerce")
    valid = den > 0
    return (num / den.where(valid)).where(valid)


@lru_cache(maxsize=8)
def _build_cached(seasons: tuple[int, ...], window: int) -> pd.DataFrame:
    players = nfl_data.load_player_stats(seasons)
    missing = [column for column in _REQUIRED_COLUMNS if column not in players.columns]
    if missing:
        raise ValueError(f"nflreadpy player stats missing columns: {missing}")

    part = players[list(_REQUIRED_COLUMNS)].copy()
    part = part[part["season_type"].isin(("REG", "POST"))]
    part = part[part["team"].notna() & part["player_id"].notna() & part["week"].notna()]
    if part.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    for column in _BASE_COLUMNS:
        part[column] = pd.to_numeric(part[column], errors="coerce").fillna(0.0)
    part["season"] = part["season"].astype(int)
    part["week"] = part["week"].astype(int)
    part = part.sort_values(["player_id", "season", "week", "game_id"], kind="stable")

    # shift(1) is the key leakage guard: the current game's row is excluded, so
    # every rolling value is known before that week's kickoff.
    rolled = part.copy()
    for column in _BASE_COLUMNS:
        rolled[f"roll_{column}"] = (
            rolled.groupby(["player_id", "season"], sort=False)[column]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
        )

    qbs = rolled[rolled["position"] == "QB"].copy()
    qbs["qb_ypa"] = _ratio(qbs, "roll_passing_yards", "roll_attempts")
    qbs["qb_epa_att"] = _ratio(qbs, "roll_passing_epa", "roll_attempts")
    qbs = qbs[qbs["roll_attempts"] > 0]
    qb = (
        qbs.sort_values(
            ["season", "week", "team", "roll_passing_yards"],
            ascending=[True, True, True, False],
            kind="stable",
        )
        .drop_duplicates(["season", "week", "team"], keep="first")
        .rename(columns={"player_name": "qb_name"})
        [["season", "week", "team", "qb_name", "qb_ypa", "qb_epa_att"]]
    )

    backs = rolled[rolled["position"] == "RB"].copy()
    backs["rb_ypc"] = _ratio(backs, "roll_rushing_yards", "roll_carries")
    backs = backs[backs["roll_carries"] > 0]
    rb = (
        backs.sort_values(
            ["season", "week", "team", "roll_rushing_yards"],
            ascending=[True, True, True, False],
            kind="stable",
        )
        .drop_duplicates(["season", "week", "team"], keep="first")
        .rename(columns={"player_name": "rb_name"})
        [["season", "week", "team", "rb_name", "rb_ypc"]]
    )

    receivers = rolled[rolled["position"].isin(("WR", "TE"))].copy()
    receivers["wr_ypr"] = _ratio(receivers, "roll_receiving_yards", "roll_receptions")
    receivers = receivers[receivers["roll_receptions"] > 0]
    receiver_order = ["season", "week", "team", "roll_receiving_yards"]
    wr = (
        receivers.sort_values(receiver_order, ascending=[True, True, True, False], kind="stable")
        .drop_duplicates(["season", "week", "team"], keep="first")
        .rename(columns={"player_name": "wr_name"})
        [["season", "week", "team", "wr_name", "wr_ypr"]]
    )
    top3 = (
        receivers.sort_values(receiver_order, ascending=[True, True, True, False], kind="stable")
        .groupby(["season", "week", "team"])
        .head(3)
        .groupby(["season", "week", "team"], as_index=False)["roll_receiving_yards"]
        .sum()
        .rename(columns={"roll_receiving_yards": "top3_rec_yards"})
    )

    out = qb.merge(rb, on=["season", "week", "team"], how="outer")
    out = out.merge(wr, on=["season", "week", "team"], how="outer")
    out = out.merge(top3, on=["season", "week", "team"], how="outer")
    for column in WEEKLY_PLAYER_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out[list(_OUTPUT_COLUMNS)].sort_values(["season", "week", "team"]).reset_index(drop=True)


def build_weekly_team_form(
    seasons: int | Iterable[int], window: int = 4
) -> pd.DataFrame:
    """One row per team-week with prior-week QB/RB/receiver leaders and rates."""
    if window < 1:
        raise ValueError("window must be at least 1")
    return _build_cached(_normalize_seasons(seasons), int(window)).copy()
