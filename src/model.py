"""Logistic-regression model for weekly NFL game winners.

The model predicts P(home team wins) from pre-game information only:

* each team's rolling EPA / defensive EPA / pace over its previous games
* rest days and a bye-week proxy (both sides plus the difference)
* one-hot team identities, which let the model learn team strength

Same-game EPA/pace is deliberately **not** used as a feature (it would leak the
result). The Kaggle closing spread is also held out of the model and used purely
as a "market" baseline, so the reported numbers show what the model learned on
its own.

Training uses the ten most recent *complete* seasons (2016-2025). Evaluation is
time-aware: a walk-forward score by season plus a final fit on all ten seasons.

Run:  python -m src.model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import data as nfl_data
from src.features import add_rolling_form, build_game_features

DEFAULT_SEASONS = tuple(range(2016, 2026))
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "logistic_regression.joblib"
METRICS_PATH = MODEL_PATH.with_suffix(".json")
FORM_COLUMNS = ("off_epa", "def_epa", "pace")
NUMERIC_FEATURES = (
    [f"{side}_{col}" for side in ("home", "away") for col in FORM_COLUMNS]
    + [f"diff_{col}" for col in FORM_COLUMNS]
    + [
        "home_rest", "away_rest", "rest_diff",
        "home_bye", "away_bye", "bye_diff",
    ]
)


def build_dataset(seasons=DEFAULT_SEASONS) -> pd.DataFrame:
    """One row per completed game with rolling form features and a home_win label."""
    games = build_game_features(seasons)
    games = add_rolling_form(games, window=3)

    scores = nfl_data.load_schedules(seasons)[
        ["game_id", "home_score", "away_score"]
    ]
    df = games.merge(scores, on="game_id", how="left", validate="one_to_one")

    played = df["home_score"].notna() & df["away_score"].notna()
    df = df[played].copy()
    # Ties have no winner; drop them (10 games across the ten seasons).
    df = df[df["home_score"] != df["away_score"]].copy()
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    return df.reset_index(drop=True)


def make_features(df: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """Build a fixed-column numeric matrix suitable for sklearn."""
    parts: dict[str, pd.Series] = {}
    for side in ("home", "away"):
        for col in FORM_COLUMNS:
            parts[f"{side}_{col}"] = pd.to_numeric(
                df[f"{side}_prev3_{col}"], errors="coerce"
            )

    for col in FORM_COLUMNS:
        parts[f"diff_{col}"] = parts[f"home_{col}"] - parts[f"away_{col}"]

    parts["home_rest"] = pd.to_numeric(df["home_rest"], errors="coerce")
    parts["away_rest"] = pd.to_numeric(df["away_rest"], errors="coerce")
    parts["rest_diff"] = parts["home_rest"] - parts["away_rest"]
    parts["home_bye"] = pd.to_numeric(df["home_had_bye"], errors="coerce")
    parts["away_bye"] = pd.to_numeric(df["away_had_bye"], errors="coerce")
    parts["bye_diff"] = parts["home_bye"] - parts["away_bye"]

    X = pd.DataFrame({name: parts[name] for name in NUMERIC_FEATURES}, index=df.index)
    for side in ("home", "away"):
        columns = [f"{side}_{team}" for team in teams]
        dummies = pd.get_dummies(df[f"{side}_team"], prefix=side, dtype=float)
        X = pd.concat([X, dummies.reindex(columns=columns, fill_value=0.0)], axis=1)
    return X


def make_pipeline(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=2000, random_state=42)),
    ])


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, proba >= 0.5)),
        "log_loss": float(log_loss(y, proba)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
    }


def train(
    seasons=DEFAULT_SEASONS,
    C: float = 1.0,
    walk_forward_from: int = 2021,
    save: bool = True,
) -> dict:
    """Fit on all `seasons`, with time-ordered validation reported first."""
    seasons = tuple(seasons)
    ds = build_dataset(seasons)
    teams = sorted(set(ds["home_team"]) | set(ds["away_team"]))
    X = make_features(ds, teams)
    y = ds["home_win"].astype(int)
    year = ds["season"].to_numpy()

    walk_forward = []
    for test_year in range(walk_forward_from, max(seasons) + 1):
        train_mask, test_mask = year < test_year, year == test_year
        if not train_mask.any() or not test_mask.any():
            continue
        season_model = make_pipeline(C).fit(X[train_mask], y[train_mask])
        walk_forward.append({
            "season": test_year,
            "games": int(test_mask.sum()),
            **evaluate(season_model, X[test_mask], y[test_mask]),
        })

    final_model = make_pipeline(C).fit(X, y)
    in_sample = evaluate(final_model, X, y)

    spread = ds["bet_home_spread"]
    has_spread = spread.notna()
    baselines = {
        "always_home": float(y.mean()),
        "market_favorite": float(
            ((spread[has_spread] > 0).astype(int) == y[has_spread]).mean()
        ),
        "market_games": int(has_spread.sum()),
    }

    artifact = {
        "model": final_model,
        "teams": teams,
        "seasons": list(seasons),
        "rolling_window": 3,
        "C": C,
        "in_sample": in_sample,
        "walk_forward": walk_forward,
        "baselines": baselines,
        "training_games": int(len(ds)),
    }

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, MODEL_PATH)
        METRICS_PATH.write_text(json.dumps({
            key: value for key, value in artifact.items() if key != "model"
        }, indent=2), encoding="utf-8")

    return artifact


def predict_games(artifact: dict, games: pd.DataFrame) -> pd.DataFrame:
    """Return home-win probabilities for games processed by `add_rolling_form`."""
    X = make_features(games, artifact["teams"])
    proba = artifact["model"].predict_proba(X)[:, 1]
    return pd.DataFrame({
        "game_id": games["game_id"].to_numpy(),
        "away_team": games["away_team"].to_numpy(),
        "home_team": games["home_team"].to_numpy(),
        "home_win_prob": proba,
    })


def load_model(path: Path = MODEL_PATH) -> dict:
    return joblib.load(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--C", type=float, default=1.0, help="L2 regularization strength")
    parser.add_argument("--no-save", action="store_true", help="fit without writing artifacts")
    args = parser.parse_args()

    artifact = train(C=args.C, save=not args.no_save)

    print(f"Trained on {artifact['training_games']:,} games "
          f"({min(artifact['seasons'])}-{max(artifact['seasons'])})")
    print(f"Features: {len(NUMERIC_FEATURES)} numeric + "
          f"{2 * len(artifact['teams'])} team indicators")
    print("\nWalk-forward validation (train on prior seasons):")
    print(f"  {'Season':<8}{'Games':>7}{'Acc':>8}{'AUC':>8}{'LogLoss':>10}{'Brier':>8}")
    for row in artifact["walk_forward"]:
        print(f"  {row['season']:<8}{row['games']:>7}{row['accuracy']:>8.3f}"
              f"{row['roc_auc']:>8.3f}{row['log_loss']:>10.3f}{row['brier']:>8.3f}")

    recent = artifact["walk_forward"][-3:]
    if recent:
        print(f"\nLast {len(recent)} seasons avg: "
              f"accuracy={np.mean([r['accuracy'] for r in recent]):.3f}, "
              f"AUC={np.mean([r['roc_auc'] for r in recent]):.3f}, "
              f"Brier={np.mean([r['brier'] for r in recent]):.3f}")
    print(f"Baselines -> always home: {artifact['baselines']['always_home']:.3f}, "
          f"market favorite: {artifact['baselines']['market_favorite']:.3f}")
    print(f"In-sample (final 10-season fit): accuracy="
          f"{artifact['in_sample']['accuracy']:.3f}, "
          f"AUC={artifact['in_sample']['roc_auc']:.3f}")
    if not args.no_save:
        print(f"\nSaved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
