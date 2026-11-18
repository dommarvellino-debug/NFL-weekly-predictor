"""Generate weekly 2026 game predictions from the trained logistic-regression model.

Predictions use only pre-game information. Team form is built from play-by-play
of already-completed games (2025 + 2026), so each upcoming game's rolling
EPA / defensive-EPA / pace features reflect games played *before* it.

The closing spread is not available for 2026 in the Kaggle dataset, so every
prediction here is the model's own probability -- there is no market line mixed
in. Predictions are most meaningful for the *next* upcoming week; forecasting
further out would rely on games that have not been played yet.

Run:  python -m src.predict            # next upcoming week
      python -m src.predict --week 12  # a specific week

Writes a machine-readable `web/data/predictions.json` for the UI and prints a
readable table to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src import data as nfl_data
from src.betting_data import load_teams
from src.features import KAGGLE_TO_NFLVERSE, add_rolling_form, build_game_features
from src.model import load_model, predict_games

SEASON = 2026
CONTEXT_SEASONS = (2025, 2026)
WEB_DATA = Path(__file__).resolve().parents[1] / "web" / "data"
PREDICTIONS_PATH = WEB_DATA / "predictions.json"
METRICS_PATH = WEB_DATA / "model-metrics.json"
FORM_COLUMNS = ("prev3_off_epa", "prev3_def_epa", "prev3_pace")

# Canonical current names keyed by nflverse abbreviation. The Kaggle teams
# table lists historical names in inconsistent order, so it's only a fallback.
CURRENT_TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos",
    "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots",
    "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers", "SF": "San Francisco 49ers",
    "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


def upcoming_week(schedule: pd.DataFrame) -> int:
    """First week that still has at least one unplayed (no score) game."""
    unplayed = schedule[schedule["home_score"].isna()]
    if unplayed.empty:
        return int(schedule["week"].max())
    return int(unplayed["week"].min())


def team_name_map() -> dict[str, str]:
    """Map nflverse team abbreviation -> current full team name."""
    names = dict(CURRENT_TEAM_NAMES)
    # Fill any abbreviation we don't know from the Kaggle lookup table.
    teams = load_teams().drop_duplicates(subset="team_id", keep="first")
    teams["nflverse_id"] = teams["team_id"].replace(KAGGLE_TO_NFLVERSE)
    for abbr, name in zip(teams["nflverse_id"], teams["team_name"]):
        names.setdefault(abbr, name)
    return names


def team_records(schedule: pd.DataFrame) -> dict[str, str]:
    """W-L record for each team from completed 2026 games."""
    played = schedule[schedule["home_score"].notna()].copy()
    wins: dict[str, int] = {}
    losses: dict[str, int] = {}
    for _, game in played.iterrows():
        home, away = game["home_team"], game["away_team"]
        home_won = game["home_score"] > game["away_score"]
        winner, loser = (home, away) if home_won else (away, home)
        wins[winner] = wins.get(winner, 0) + 1
        losses[loser] = losses.get(loser, 0) + 1
    teams = set(played["home_team"]) | set(played["away_team"])
    return {team: f"{wins.get(team, 0)}-{losses.get(team, 0)}" for team in teams}


def _confidence(prob: float) -> str:
    edge = abs(prob - 0.5)
    if edge >= 0.20:
        return "high"
    if edge >= 0.10:
        return "moderate"
    return "low"


def generate_week_predictions(
    week: int | None = None,
    season: int = SEASON,
    artifact: dict | None = None,
) -> pd.DataFrame:
    """Return one row per game in `week` with home-win probability + metadata."""
    artifact = artifact if artifact is not None else load_model()
    schedule = nfl_data.load_schedules(season)
    if week is None:
        week = upcoming_week(schedule)

    # Build features across the context seasons so rolling form carries over
    # the offseason into the early weeks of the target season.
    context = tuple(s for s in CONTEXT_SEASONS if s <= season)
    games = build_game_features(context, include_betting=False)
    games = add_rolling_form(games, window=artifact["rolling_window"])

    week_games = games[(games["season"] == season) & (games["week"] == week)].copy()
    if week_games.empty:
        raise ValueError(f"No games found for {season} week {week}")

    picks = predict_games(artifact, week_games)
    picks["week"] = week

    # Kickoff / venue metadata and the actual pre-game inputs used by the model.
    schedule_meta = schedule.loc[
        schedule["week"] == week,
        ["game_id", "gameday", "weekday", "gametime", "stadium"],
    ]
    picks = picks.merge(schedule_meta, on="game_id", how="left", validate="one_to_one")
    picks["gameday"] = picks["gameday"].astype(str)

    form_cols = [f"{side}_{col}" for side in ("home", "away") for col in FORM_COLUMNS]
    form_cols += ["home_rest", "away_rest", "home_had_bye", "away_had_bye"]
    form = week_games[["game_id"] + form_cols].copy()
    picks = picks.merge(form, on="game_id", how="left", validate="one_to_one")

    names = team_name_map()
    records = team_records(schedule)
    picks["home_name"] = picks["home_team"].map(names)
    picks["away_name"] = picks["away_team"].map(names)
    picks["home_record"] = picks["home_team"].map(records).fillna("0-0")
    picks["away_record"] = picks["away_team"].map(records).fillna("0-0")
    picks["away_win_prob"] = 1.0 - picks["home_win_prob"]
    picks["predicted_winner"] = picks.apply(
        lambda row: row["home_team"] if row["home_win_prob"] >= 0.5 else row["away_team"],
        axis=1,
    )
    picks["confidence"] = picks["home_win_prob"].apply(_confidence)
    # Rank by how confident the model is in the predicted winner, not by raw
    # home probability (which would scatter away-team picks down the table).
    picks["_margin"] = (picks["home_win_prob"] - 0.5).abs()
    return picks.sort_values("_margin", ascending=False).drop(columns="_margin").reset_index(drop=True)


def _side_features(row, side: str) -> dict:
    return {
        "off_epa": round(float(getattr(row, f"{side}_prev3_off_epa")), 2),
        "def_epa": round(float(getattr(row, f"{side}_prev3_def_epa")), 2),
        "pace": round(float(getattr(row, f"{side}_prev3_pace")), 1),
        "rest_days": int(getattr(row, f"{side}_rest")),
        "bye": bool(getattr(row, f"{side}_had_bye")),
    }


def to_json(
    picks: pd.DataFrame, week: int, season: int = SEASON, artifact: dict | None = None
) -> dict:
    """Shape the prediction table for the static UI / external consumers."""
    artifact = artifact if artifact is not None else load_model()
    recent = artifact["walk_forward"][-3:]
    meta = {
        "type": "logistic_regression",
        "seasons": artifact["seasons"],
        "training_games": artifact["training_games"],
        "features": artifact["feature_count"],
        "walk_forward_accuracy": round(sum(r["accuracy"] for r in recent) / len(recent), 4),
        "walk_forward_auc": round(sum(r["roc_auc"] for r in recent) / len(recent), 4),
    }
    return {
        "season": season,
        "week": week,
        "model": meta,
        "note": "Model probabilities only; no betting line available for 2026.",
        "games": [
            {
                "game_id": row.game_id,
                "gameday": str(row.gameday),
                "weekday": str(row.weekday),
                "gametime": str(row.gametime),
                "stadium": str(row.stadium),
                "home": {
                    "abbr": row.home_team,
                    "name": row.home_name,
                    "record": row.home_record,
                    "win_prob": round(float(row.home_win_prob), 4),
                    "features": _side_features(row, "home"),
                },
                "away": {
                    "abbr": row.away_team,
                    "name": row.away_name,
                    "record": row.away_record,
                    "win_prob": round(float(row.away_win_prob), 4),
                    "features": _side_features(row, "away"),
                },
                "predicted_winner": row.predicted_winner,
                "confidence": row.confidence,
                "edge_pp": round((abs(float(row.home_win_prob) - 0.5)) * 100, 1),
            }
            for row in picks.itertuples(index=False)
        ],
    }


def write_model_metrics(artifact: dict, path: Path = METRICS_PATH) -> Path:
    """Export training metadata + the out-of-fold reliability curve for the UI."""
    path.parent.mkdir(parents=True, exist_ok=True)
    recent = artifact["walk_forward"][-3:]
    payload = {
        "seasons": artifact["seasons"],
        "training_games": artifact["training_games"],
        "features": artifact["feature_count"],
        "num_teams": artifact["num_teams"],
        "C": artifact["C"],
        "rolling_window": artifact["rolling_window"],
        "in_sample": artifact["in_sample"],
        "walk_forward": artifact["walk_forward"],
        "calibration": artifact["calibration"],
        "baselines": artifact["baselines"],
        "recent_average": {
            "accuracy": round(sum(r["accuracy"] for r in recent) / len(recent), 4),
            "roc_auc": round(sum(r["roc_auc"] for r in recent) / len(recent), 4),
            "brier": round(sum(r["brier"] for r in recent) / len(recent), 4),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def print_table(picks: pd.DataFrame, week: int, season: int = SEASON) -> None:
    print(f"\n{season} Week {week} predictions ({len(picks)} games)\n")
    header = f"  {'Matchup':<26}{'Prob':>7}  {'Pick':<5}  Confidence"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in picks.itertuples(index=False):
        matchup = f"{row.away_team} @ {row.home_team}"
        prob = row.home_win_prob if row.predicted_winner == row.home_team else row.away_win_prob
        print(f"  {matchup:<26}{prob:>6.1%}  {row.predicted_winner:<5}  {row.confidence}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week", type=int, default=None, help="week to predict (default: next unplayed)")
    parser.add_argument("--season", type=int, default=SEASON, help="season to predict")
    parser.add_argument("--json", type=Path, default=PREDICTIONS_PATH, help="output JSON path")
    parser.add_argument("--no-write", action="store_true", help="print only; do not write JSON")
    args = parser.parse_args()

    artifact = load_model()
    picks = generate_week_predictions(week=args.week, season=args.season, artifact=artifact)
    week = int(picks["week"].iloc[0])
    print_table(picks, week, args.season)

    if not args.no_write:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(to_json(picks, week, args.season, artifact), indent=2), encoding="utf-8"
        )
        metrics_path = write_model_metrics(artifact)
        print(f"\nSaved predictions -> {args.json}")
        print(f"Saved model metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
