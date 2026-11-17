"""Data loading helpers.

nflreadpy returns Polars DataFrames; these wrappers convert them to pandas so
the rest of the pipeline (feature engineering, scikit-learn, seaborn) can use a
single dataframe library.
"""

from __future__ import annotations

from typing import Iterable

import nflreadpy as nfl
import pandas as pd


def load_schedules(seasons: int | Iterable[int]) -> pd.DataFrame:
    """Load nflverse game schedules for one or more seasons as a pandas DataFrame."""
    return nfl.load_schedules(seasons=seasons).to_pandas()


def load_player_stats(seasons: int | Iterable[int]) -> pd.DataFrame:
    """Load nflverse player season stats as a pandas DataFrame."""
    return nfl.load_player_stats(seasons=seasons).to_pandas()


def load_team_stats(seasons: int | Iterable[int]) -> pd.DataFrame:
    """Load nflverse team season stats as a pandas DataFrame."""
    return nfl.load_team_stats(seasons=seasons).to_pandas()


def load_pbp(seasons: int | Iterable[int]) -> pd.DataFrame:
    """Load nflverse play-by-play as a pandas DataFrame (EPA, plays, teams)."""
    return nfl.load_pbp(seasons=seasons).to_pandas()
