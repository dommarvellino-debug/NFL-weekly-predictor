"""Game-level features for the weekly predictor.

Feature availability by source
------------------------------
| Feature          | Source | How |
| ---------------- | ------ | --- |
| Offensive EPA    | nflreadpy PBP | sum/mean of `epa` where `posteam` = offense |
| Defensive EPA    | nflreadpy PBP | sum/mean of `epa` where `defteam` = defense (EPA allowed) |
| Pace             | nflreadpy PBP | number of offensive plays per game |
| Rest days        | nflreadpy schedules | `home_rest` / `away_rest` (days since last game) |
| Bye week         | derived | rest >= 12 days is used as a bye proxy |
| Player form      | Kaggle player stats | prior-season QB/RB/receiver leaders + top-3 receiving yards |
| Betting lines    | Kaggle | spread + over/under, joined on season/date/teams |

The PBP aggregates describe the *whole game*. For prediction, use
:func:`add_rolling_form` so a game's features only use games played **before**
it (avoids label leakage).
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from src import data as nfl_data
from src.betting_data import load_scores as load_kaggle_scores
from src.player_stats import (
    PLAYER_COLUMNS,
    PLAYER_NAME_COLUMNS,
    build_team_season_player_stats,
)

# Kaggle's team IDs differ from nflverse for these two franchises.
KAGGLE_TO_NFLVERSE = {"LAR": "LA", "LVR": "LV"}


def load_team_game_stats(seasons: int | Iterable[int]) -> pd.DataFrame:
    """Per team-game offensive EPA, defensive EPA and pace from play-by-play."""
    pbp = nfl_data.load_pbp(seasons)
    plays = pbp[pbp["epa"].notna() & pbp["posteam"].notna() & pbp["defteam"].notna()]

    offense = (
        plays.groupby(["game_id", "posteam"])
        .agg(off_epa=("epa", "sum"), off_epa_per_play=("epa", "mean"), pace=("epa", "size"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    defense = (
        plays.groupby(["game_id", "defteam"])
        .agg(def_epa=("epa", "sum"), def_epa_per_play=("epa", "mean"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    return offense.merge(defense, on=["game_id", "team"], how="outer")


def _attach_side(base: pd.DataFrame, stats: pd.DataFrame, side: str) -> pd.DataFrame:
    """Attach a team-game stat table as `home_*` or `away_*` columns."""
    lookup = stats.set_index(["game_id", "team"])
    keys = pd.MultiIndex.from_frame(
        base[["game_id", f"{side}_team"]].rename(columns={f"{side}_team": "team"})
    )
    values = lookup.reindex(keys).reset_index(drop=True)
    values.columns = [f"{side}_{col}" for col in values.columns]
    return pd.concat([base.reset_index(drop=True), values], axis=1)


def _attach_betting(base: pd.DataFrame) -> pd.DataFrame:
    """Join Kaggle spread/over-under/weather onto nflverse games."""
    kaggle = load_kaggle_scores()
    keep = [
        "schedule_season", "schedule_date", "home_id", "away_id",
        "home_spread", "over_under_line",
        "weather_temperature", "weather_wind_mph", "stadium_neutral",
    ]
    betting = kaggle[keep].copy()
    betting["home_id"] = betting["home_id"].replace(KAGGLE_TO_NFLVERSE)
    betting["away_id"] = betting["away_id"].replace(KAGGLE_TO_NFLVERSE)
    betting["gameday"] = betting["schedule_date"].dt.strftime("%Y-%m-%d")
    betting = betting.rename(columns={
        "schedule_season": "season",
        "home_spread": "bet_home_spread",
        "over_under_line": "bet_over_under",
        "weather_temperature": "bet_temp",
        "weather_wind_mph": "bet_wind",
        "stadium_neutral": "bet_neutral",
    })
    betting = betting[[
        "season", "gameday", "home_id", "away_id",
        "bet_home_spread", "bet_over_under", "bet_temp", "bet_wind", "bet_neutral",
    ]].drop_duplicates(subset=["season", "gameday", "home_id", "away_id"])

    merged = base.merge(
        betting,
        left_on=["season", "gameday", "home_team", "away_team"],
        right_on=["season", "gameday", "home_id", "away_id"],
        how="left",
    )
    return merged.drop(columns=["home_id", "away_id"])


def add_player_features(games: pd.DataFrame) -> pd.DataFrame:
    """Attach prior-season player leaders for both teams.

    Only season ``S - 1`` is joined to a game in season ``S``. That keeps the
    player inputs available before kickoff and avoids using a team's current
    season statistics to predict its current-game result.
    """
    stats = build_team_season_player_stats().copy()
    stats["season"] = stats["season"] + 1  # season N stats are available in N+1
    lookup = stats[["season", "team"] + list(PLAYER_COLUMNS) + list(PLAYER_NAME_COLUMNS)]
    lookup = lookup.set_index(["season", "team"])

    out = games.copy().reset_index(drop=True)
    columns = list(PLAYER_COLUMNS) + list(PLAYER_NAME_COLUMNS)
    for side in ("home", "away"):
        keys = pd.MultiIndex.from_frame(
            out[["season", f"{side}_team"]].rename(columns={f"{side}_team": "team"})
        )
        values = lookup.reindex(keys).reset_index(drop=True)
        values.columns = [f"{side}_player_{col}" for col in columns]
        out = pd.concat([out, values], axis=1)
    return out


def build_game_features(
    seasons: int | Iterable[int], include_betting: bool = True
) -> pd.DataFrame:
    """One row per game with EPA, pace, rest, bye, and prior-season player form."""
    schedule = nfl_data.load_schedules(seasons)
    team_game = load_team_game_stats(seasons)

    base = schedule[[
        "game_id", "season", "game_type", "week", "gameday",
        "home_team", "away_team", "home_rest", "away_rest",
    ]].copy()

    base = _attach_side(base, team_game, "home")
    base = _attach_side(base, team_game, "away")

    # A long layoff is a reliable proxy for a bye week.
    for side in ("home", "away"):
        base[f"{side}_had_bye"] = (base[f"{side}_rest"] >= 12).astype(int)

    if include_betting:
        base = _attach_betting(base)
    return add_player_features(base)


def add_rolling_form(games: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Add each team's rolling *prior-game* EPA/pace (leakage-free form)."""
    columns = ["off_epa", "def_epa", "pace"]
    frames = []
    for side in ("home", "away"):
        part = games[["game_id", "gameday", f"{side}_team"] +
                     [f"{side}_{col}" for col in columns]].copy()
        part.columns = ["game_id", "gameday", "team"] + columns
        frames.append(part)

    long = pd.concat(frames, ignore_index=True).sort_values(["team", "gameday"])
    rolling_cols = []
    for col in columns:
        new_col = f"prev{window}_{col}"
        long[new_col] = (
            long.groupby("team")[col]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        rolling_cols.append(new_col)

    out = games.copy()
    lookup = long.set_index(["game_id", "team"])[rolling_cols]
    for side in ("home", "away"):
        keys = pd.MultiIndex.from_frame(
            out[["game_id", f"{side}_team"]].rename(columns={f"{side}_team": "team"})
        )
        values = lookup.reindex(keys).reset_index(drop=True)
        values.columns = [f"{side}_{col}" for col in rolling_cols]
        out = pd.concat([out.reset_index(drop=True), values], axis=1)
    return out
