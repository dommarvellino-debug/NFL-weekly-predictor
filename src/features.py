"""Game-level features for the weekly predictor.

Feature availability by source
------------------------------
| Feature          | Source | How |
| ---------------- | ------ | --- |
| Offensive EPA    | nflreadpy PBP | sum/mean of `epa` where `posteam` = offense |
| Defensive EPA    | nflreadpy PBP | sum/mean of `epa` where `defteam` = defense (EPA allowed) |
| Success rate     | nflreadpy PBP | mean of `success` on offensive plays |
| Pace             | nflreadpy PBP | number of offensive plays per game |
| Rest days        | nflreadpy schedules | `home_rest` / `away_rest` (days since last game) |
| Bye week         | derived | rest >= 12 days is used as a bye proxy |
| Player form      | Kaggle player stats | prior-season QB/RB/receiver leaders + top-3 receiving yards |
| Weekly form      | nflreadpy player stats | prior-week QB/RB/receiver leaders, joined by team-week |
| Betting lines    | Kaggle | spread + over/under, joined on season/date/teams |

The PBP aggregates describe the *whole game*. For prediction, use
:func:`add_rolling_form` so a game's features only use games played **before**
it (avoids label leakage). Weekly player form uses rolling player sums that also
shift by one game.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src import data as nfl_data
from src.betting_data import load_scores as load_kaggle_scores
from src.player_stats import (
    PLAYER_COLUMNS,
    PLAYER_NAME_COLUMNS,
    build_team_season_player_stats,
)
from src.weekly_players import (
    WEEKLY_PLAYER_COLUMNS,
    WEEKLY_PLAYER_NAME_COLUMNS,
    build_weekly_team_form,
)

# Kaggle's team IDs differ from nflverse for these two franchises.
KAGGLE_TO_NFLVERSE = {"LAR": "LA", "LVR": "LV"}

# Rolling team-game columns before the `prevN_` prefix is applied.
ROLLING_BASE_COLUMNS = (
    "off_epa",
    "def_epa",
    "pace",
    "off_epa_per_play",
    "def_epa_per_play",
    "success_rate",
)
# Derived from both teams' rolling rates; also stored with the `prevN_` prefix.
MATCHUP_COLUMNS = ("adj_off_epa", "adj_def_epa")


def load_team_game_stats(seasons: int | Iterable[int]) -> pd.DataFrame:
    """Per team-game EPA rates, success rate, defensive EPA and pace."""
    pbp = nfl_data.load_pbp(seasons)
    plays = pbp[pbp["epa"].notna() & pbp["posteam"].notna() & pbp["defteam"].notna()]

    offense = (
        plays.groupby(["game_id", "posteam"])
        .agg(
            off_epa=("epa", "sum"),
            off_epa_per_play=("epa", "mean"),
            success_rate=("success", "mean"),
            pace=("epa", "size"),
        )
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


def add_weekly_player_features(
    games: pd.DataFrame, window: int = 4
) -> pd.DataFrame:
    """Attach the latest prior-week player form for both teams.

    A merge_asof carries the most recent team-week row forward. For a played
    game the exact week is used, but the rolling player sums already exclude
    that game. For an upcoming week with no player rows yet (e.g. week 11 after
    week 10 is complete), the latest prior week is used instead.
    """
    stats = build_weekly_team_form(games["season"].unique(), window=window)
    if stats.empty:
        return games
    columns = list(WEEKLY_PLAYER_NAME_COLUMNS) + list(WEEKLY_PLAYER_COLUMNS)
    right = stats[["season", "week", "team"] + columns].copy()
    right["season"] = right["season"].astype("int64")
    right["week"] = right["week"].astype("int64")
    right = right.sort_values("week")

    out = games.copy().reset_index(drop=True)
    for side in ("home", "away"):
        left = out[["season", "week", f"{side}_team"]].copy()
        left = left.rename(columns={f"{side}_team": "team"})
        left["season"] = left["season"].astype("int64")
        left["week"] = left["week"].astype("int64")
        left["_row"] = np.arange(len(left))
        merged = pd.merge_asof(
            left.sort_values("week"),
            right,
            on="week",
            by=["season", "team"],
            direction="backward",
            allow_exact_matches=True,
        ).sort_values("_row")
        values = merged[columns].reset_index(drop=True)
        values.columns = [f"{side}_weekly_{col}" for col in columns]
        out = pd.concat([out, values], axis=1)
    return out


def build_game_features(
    seasons: int | Iterable[int], include_betting: bool = True,
    weekly_window: int = 4,
) -> pd.DataFrame:
    """One row per game with EPA rates, rest, prior- and weekly-season player form."""
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
    base = add_player_features(base)
    return add_weekly_player_features(base, window=weekly_window)


def add_rolling_form(games: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Add prior-game EPA/pace/success and opponent-adjusted EPA (leakage-free)."""
    prefix = f"prev{window}_"
    frames = []
    for side in ("home", "away"):
        part = games[["game_id", "gameday", f"{side}_team"] +
                     [f"{side}_{col}" for col in ROLLING_BASE_COLUMNS]].copy()
        part.columns = ["game_id", "gameday", "team"] + list(ROLLING_BASE_COLUMNS)
        frames.append(part)

    long = pd.concat(frames, ignore_index=True).sort_values(["team", "gameday"])
    rolling_cols = []
    for col in ROLLING_BASE_COLUMNS:
        new_col = f"{prefix}{col}"
        long[new_col] = (
            long.groupby("team")[col]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        rolling_cols.append(new_col)

    lookup = long.set_index(["game_id", "team"])[rolling_cols]
    out = games.copy().reset_index(drop=True)
    for side in ("home", "away"):
        keys = pd.MultiIndex.from_frame(
            out[["game_id", f"{side}_team"]].rename(columns={f"{side}_team": "team"})
        )
        values = lookup.reindex(keys).reset_index(drop=True)
        values.columns = [f"{side}_{col}" for col in rolling_cols]
        out = pd.concat([out, values], axis=1)

    # Team offense together with the opponent's defense (and vice versa). Both
    # halves come from games before this one, so no same-game outcome leaks in.
    for side, opponent in (("home", "away"), ("away", "home")):
        own_off = f"{side}_{prefix}off_epa_per_play"
        own_def = f"{side}_{prefix}def_epa_per_play"
        opp_off = f"{opponent}_{prefix}off_epa_per_play"
        opp_def = f"{opponent}_{prefix}def_epa_per_play"
        out[f"{side}_{prefix}adj_off_epa"] = out[own_off] + out[opp_def]
        out[f"{side}_{prefix}adj_def_epa"] = out[own_def] + out[opp_off]
    return out
