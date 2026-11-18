"""Kaggle player-stat loader and team-season aggregates.

Dataset: https://www.kaggle.com/datasets/philiphyde1/nfl-stats-1999-2022
Downloaded with kagglehub (the current version contains seasons 2012-2025).

Only the compact yearly file is read. From it we derive, for every team and
season, the leading QB / RB / receiver plus a receiving-depth measure. These are
joined to games as *prior-season* features, so a game never sees player stats
from its own season (which would leak future information).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

DATASET_HANDLE = "philiphyde1/nfl-stats-1999-2022"
YEARLY_FILE = "yearly_player_stats_offense.csv"

# Older Kaggle snapshots used different codes for these franchises.
KAGGLE_TO_NFLVERSE = {"LAR": "LA", "LVR": "LV"}

PLAYER_COLUMNS = (
    "qb_rating",
    "qb_ypa",
    "rb_ypc",
    "wr_ypr",
    "top3_rec_yards",
)
PLAYER_NAME_COLUMNS = ("qb_name", "rb_name", "wr_name")

_USECOLS = (
    "player_name", "position_group", "team", "season", "season_type",
    "pass_attempts", "passing_yards", "ypa", "passer_rating",
    "rush_attempts", "rushing_yards", "ypc",
    "targets", "receptions", "receiving_yards", "ypr",
)


@lru_cache(maxsize=1)
def ensure_downloaded() -> Path:
    """Download (once) and return the cached Kaggle dataset directory."""
    import kagglehub  # lazy: normal imports don't require Kaggle access

    return Path(kagglehub.dataset_download(DATASET_HANDLE))


@lru_cache(maxsize=1)
def load_yearly_player_stats() -> pd.DataFrame:
    """Return the yearly offensive player table with only the columns we need."""
    path = ensure_downloaded() / YEARLY_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing {YEARLY_FILE} in Kaggle dataset: {path}")
    return pd.read_csv(path, usecols=list(_USECOLS), low_memory=False)


def _leader(frame: pd.DataFrame, groups: tuple[str, ...], sort_col: str) -> pd.DataFrame:
    """One row per team-season: the player with the highest `sort_col`."""
    part = frame[frame["position_group"].isin(groups)].copy()
    part[sort_col] = pd.to_numeric(part[sort_col], errors="coerce").fillna(0.0)
    part = part[part[sort_col] > 0]
    return (
        part.sort_values(["season", "team", sort_col], ascending=[True, True, False])
        .drop_duplicates(["season", "team"], keep="first")
    )


@lru_cache(maxsize=1)
def build_team_season_player_stats() -> pd.DataFrame:
    """Aggregate player leaders into one row per team-season (regular season)."""
    players = load_yearly_player_stats()
    players = players[players["season_type"] == "REG"].copy()
    players["team"] = players["team"].replace(KAGGLE_TO_NFLVERSE)
    players["season"] = pd.to_numeric(players["season"], errors="coerce")

    qbs = _leader(players, ("QB",), "pass_attempts")
    qbs = qbs.rename(columns={
        "player_name": "qb_name",
        "passer_rating": "qb_rating",
        "ypa": "qb_ypa",
    })[["season", "team", "qb_name", "qb_rating", "qb_ypa"]]

    backs = _leader(players, ("RB",), "rushing_yards")
    backs = backs.rename(columns={
        "player_name": "rb_name",
        "ypc": "rb_ypc",
    })[["season", "team", "rb_name", "rb_ypc"]]

    receivers = players[players["position_group"].isin(("WR", "TE"))].copy()
    receivers["receiving_yards"] = pd.to_numeric(
        receivers["receiving_yards"], errors="coerce"
    ).fillna(0.0)
    top_receiver = (
        receivers.sort_values(["season", "team", "receiving_yards"], ascending=[True, True, False])
        .drop_duplicates(["season", "team"], keep="first")
        .rename(columns={"player_name": "wr_name", "ypr": "wr_ypr"})
        [["season", "team", "wr_name", "wr_ypr"]]
    )
    top3 = (
        receivers.sort_values(["season", "team", "receiving_yards"], ascending=[True, True, False])
        .groupby(["season", "team"])
        .head(3)
        .groupby(["season", "team"], as_index=False)["receiving_yards"]
        .sum()
        .rename(columns={"receiving_yards": "top3_rec_yards"})
    )

    out = qbs.merge(backs, on=["season", "team"], how="left")
    out = out.merge(top_receiver, on=["season", "team"], how="left")
    out = out.merge(top3, on=["season", "team"], how="left")

    for col in PLAYER_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["season"])
    out["season"] = out["season"].astype(int)
    return out.sort_values(["season", "team"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def player_stats_coverage() -> dict:
    """Metadata describing what the Kaggle player source actually contains."""
    stats = build_team_season_player_stats()
    return {
        "dataset": DATASET_HANDLE,
        "file": YEARLY_FILE,
        "seasons": [int(stats["season"].min()), int(stats["season"].max())],
        "team_seasons": int(len(stats)),
        "teams": int(stats["team"].nunique()),
        "features": list(PLAYER_COLUMNS),
    }
