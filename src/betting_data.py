"""Loader for the Kaggle 'NFL Scores & Betting Data' dataset.

Dataset: https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data

The raw CSVs live in ``data/raw``. If they are missing, :func:`ensure_downloaded`
pulls the latest version with kagglehub and caches a copy there.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

DATASET_HANDLE = "tobycrabtree/nfl-scores-and-betting-data"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
SCORES_FILE = "spreadspoke_scores.csv"
TEAMS_FILE = "nfl_teams.csv"


def ensure_downloaded(raw_dir: Path = RAW_DIR) -> Path:
    """Ensure the Kaggle CSVs are available locally; download them if needed."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if (raw_dir / SCORES_FILE).exists() and (raw_dir / TEAMS_FILE).exists():
        return raw_dir

    import kagglehub  # imported lazily so normal runs don't require it

    cached = Path(kagglehub.dataset_download(DATASET_HANDLE))
    for name in (SCORES_FILE, TEAMS_FILE):
        source = cached / name
        if source.exists():
            shutil.copy2(source, raw_dir / name)
    return raw_dir


def load_scores(raw_dir: Path = RAW_DIR, download: bool = True) -> pd.DataFrame:
    """Return game-level scores + betting lines as a pandas DataFrame.

    Adds derived columns:
      home_id / away_id  - team abbreviations (joined from nfl_teams.csv)
      home_win           - 1 if the home team won
      home_margin        - home score minus away score
      total_points       - combined score
      home_spread        - betting spread from the home team's perspective
                           (positive = home favored, negative = home underdog)
    """
    if download:
        ensure_downloaded(raw_dir)

    games = pd.read_csv(Path(raw_dir) / SCORES_FILE)
    teams = pd.read_csv(Path(raw_dir) / TEAMS_FILE)

    name_to_id = teams.set_index("team_name")["team_id"]
    games["home_id"] = games["team_home"].map(name_to_id)
    games["away_id"] = games["team_away"].map(name_to_id)

    games["schedule_date"] = pd.to_datetime(games["schedule_date"])
    games["home_win"] = (games["score_home"] > games["score_away"]).astype(int)
    games["home_margin"] = games["score_home"] - games["score_away"]
    games["total_points"] = games["score_home"] + games["score_away"]

    spread = games["spread_favorite"]
    favorite = games["team_favorite_id"]
    # spread_favorite is negative for the favorite; flip it when the home team
    # is the favorite so the number always reads from the home team's view.
    games["home_spread"] = 0.0
    home_fav = favorite.eq(games["home_id"])
    away_fav = favorite.eq(games["away_id"])
    games.loc[home_fav, "home_spread"] = -spread[home_fav]
    games.loc[away_fav, "home_spread"] = spread[away_fav]
    games.loc[favorite.eq("PICK") | favorite.isna(), "home_spread"] = 0.0

    return games


def load_teams(raw_dir: Path = RAW_DIR, download: bool = True) -> pd.DataFrame:
    """Return the team name -> abbreviation mapping table."""
    if download:
        ensure_downloaded(raw_dir)
    return pd.read_csv(Path(raw_dir) / TEAMS_FILE)
