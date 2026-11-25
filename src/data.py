"""Data loading helpers.

nflreadpy returns Polars DataFrames; these wrappers convert them to pandas so
the rest of the pipeline (feature engineering, scikit-learn, seaborn) can use a
single dataframe library.

The nflverse ``schedules`` release briefly publishes only ``games.csv`` while
its parquet asset is being rebuilt. Pinned nflreadpy versions request the
parquet file unconditionally, so :func:`load_schedules` falls back to the CSV
asset when that download fails.
"""

from __future__ import annotations

from typing import Iterable

import nflreadpy as nfl
import pandas as pd

_RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"
SCHEDULES_CSV_URL = f"{_RELEASES}/schedules/games.csv"


def _season_list(seasons: int | Iterable[int]) -> list[int]:
    """Normalize a single season or an iterable of seasons to a list."""
    if isinstance(seasons, int):
        return [seasons]
    return list(seasons)


def load_schedules(seasons: int | Iterable[int]) -> pd.DataFrame:
    """Load nflverse game schedules for one or more seasons as a pandas DataFrame.

    Falls back to the published CSV when the parquet asset is temporarily
    unavailable, filtering to the requested seasons so the result matches
    ``nflreadpy.load_schedules``.
    """
    try:
        return nfl.load_schedules(seasons=seasons).to_pandas()
    except ConnectionError:
        csv = pd.read_csv(SCHEDULES_CSV_URL, low_memory=False)
        wanted = _season_list(seasons)
        return csv[csv["season"].isin(wanted)].reset_index(drop=True)


def load_player_stats(seasons: int | Iterable[int]) -> pd.DataFrame:
    """Load nflverse player season stats as a pandas DataFrame."""
    return nfl.load_player_stats(seasons=seasons).to_pandas()


def load_team_stats(seasons: int | Iterable[int]) -> pd.DataFrame:
    """Load nflverse team season stats as a pandas DataFrame."""
    return nfl.load_team_stats(seasons=seasons).to_pandas()


def load_pbp(seasons: int | Iterable[int]) -> pd.DataFrame:
    """Load nflverse play-by-play as a pandas DataFrame (EPA, plays, teams)."""
    return nfl.load_pbp(seasons=seasons).to_pandas()
