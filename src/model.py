"""Logistic-regression model for weekly NFL game winners.

The model predicts P(home team wins) from pre-game information only:

* rolling EPA totals, pace and opponent-adjusted EPA (built from each team's
  EPA/play and the opponent's EPA allowed per play) over previous games
* rest days and a bye-week proxy (both sides plus the difference)
* prior-season player leaders (Kaggle) and current-season prior-week player
  leaders/rates (nflreadpy) for both teams
* one-hot team identities, which let the model learn team strength

Same-game EPA/pace and current-game player stats are deliberately **not** used
as features (they would leak the result). The Kaggle closing spread is also held
out of the model and used purely as a "market" baseline.

Training uses the ten most recent *complete* seasons (2016-2025). Evaluation is
time-aware: an outer walk-forward score by season, an inner walk-forward search
for the L2 strength ``C``, and temperature calibration fitted only on earlier
out-of-fold predictions. A final artifact is refit on all ten seasons.

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
from src.features import MATCHUP_COLUMNS, add_rolling_form, build_game_features
from src.player_stats import PLAYER_COLUMNS, player_stats_coverage
from src.weekly_players import WEEKLY_PLAYER_COLUMNS

DEFAULT_SEASONS = tuple(range(2016, 2026))
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "logistic_regression.joblib"
METRICS_PATH = MODEL_PATH.with_suffix(".json")

ROLLING_WINDOW = 3
WEEKLY_WINDOW = 4
INNER_WALK_FORWARD_FROM = 2019
C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)
T_GRID = (0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5)

# EPA/play and success rate are computed by features.add_rolling_form and
# exported to the UI; the adjusted-EPA columns are the model inputs built from
# EPA/play. Raw EPA totals were kept after ablation beat adding the rates alone.
FORM_COLUMNS = ("off_epa", "def_epa", "pace") + MATCHUP_COLUMNS
# QB EPA/attempt is exported for insight but is too noisy as a model input.
WEEKLY_MODEL_COLUMNS = tuple(
    col for col in WEEKLY_PLAYER_COLUMNS if col != "qb_epa_att"
)
NUMERIC_FEATURES = (
    [f"{side}_{col}" for side in ("home", "away") for col in FORM_COLUMNS]
    + [f"diff_{col}" for col in FORM_COLUMNS]
    + [
        "home_rest", "away_rest", "rest_diff",
        "home_bye", "away_bye", "bye_diff",
    ]
    + [f"{side}_player_{col}" for side in ("home", "away") for col in PLAYER_COLUMNS]
    + [f"diff_player_{col}" for col in PLAYER_COLUMNS]
    + [
        f"{side}_weekly_{col}"
        for side in ("home", "away")
        for col in WEEKLY_MODEL_COLUMNS
    ]
    + [f"diff_weekly_{col}" for col in WEEKLY_MODEL_COLUMNS]
)


def build_dataset(seasons=DEFAULT_SEASONS) -> pd.DataFrame:
    """One row per completed game with rolling features and a home_win label."""
    games = build_game_features(seasons, weekly_window=WEEKLY_WINDOW)
    games = add_rolling_form(games, window=ROLLING_WINDOW)

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
    prefix = f"prev{ROLLING_WINDOW}_"
    for side in ("home", "away"):
        for col in FORM_COLUMNS:
            parts[f"{side}_{col}"] = pd.to_numeric(
                df[f"{side}_{prefix}{col}"], errors="coerce"
            )

    for col in FORM_COLUMNS:
        parts[f"diff_{col}"] = parts[f"home_{col}"] - parts[f"away_{col}"]

    parts["home_rest"] = pd.to_numeric(df["home_rest"], errors="coerce")
    parts["away_rest"] = pd.to_numeric(df["away_rest"], errors="coerce")
    parts["rest_diff"] = parts["home_rest"] - parts["away_rest"]
    parts["home_bye"] = pd.to_numeric(df["home_had_bye"], errors="coerce")
    parts["away_bye"] = pd.to_numeric(df["away_had_bye"], errors="coerce")
    parts["bye_diff"] = parts["home_bye"] - parts["away_bye"]

    for side in ("home", "away"):
        for col in PLAYER_COLUMNS:
            parts[f"{side}_player_{col}"] = pd.to_numeric(
                df[f"{side}_player_{col}"], errors="coerce"
            )
    for col in PLAYER_COLUMNS:
        parts[f"diff_player_{col}"] = (
            parts[f"home_player_{col}"] - parts[f"away_player_{col}"]
        )

    for side in ("home", "away"):
        for col in WEEKLY_MODEL_COLUMNS:
            parts[f"{side}_weekly_{col}"] = pd.to_numeric(
                df[f"{side}_weekly_{col}"], errors="coerce"
            )
    for col in WEEKLY_MODEL_COLUMNS:
        parts[f"diff_weekly_{col}"] = (
            parts[f"home_weekly_{col}"] - parts[f"away_weekly_{col}"]
        )

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


def score_predictions(y: pd.Series | np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Score probabilities from a fixed label vector."""
    labels = np.asarray(y)
    probs = np.asarray(proba, dtype=float)
    return {
        "accuracy": float(accuracy_score(labels, probs >= 0.5)),
        "log_loss": float(log_loss(labels, probs)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "brier": float(brier_score_loss(labels, probs)),
    }


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    return score_predictions(y, model.predict_proba(X)[:, 1])


def calibrate_proba(proba: np.ndarray | float, temperature: float) -> np.ndarray:
    """Sharpen/soften probabilities with `sigmoid(logit(p) / T)`."""
    p = np.asarray(proba, dtype=float)
    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)) / float(temperature or 1.0)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def fit_temperature(
    labels: list[int] | np.ndarray,
    proba: list[float] | np.ndarray,
    grid: tuple[float, ...] = T_GRID,
) -> float:
    """Choose the temperature that minimizes log loss on earlier OOF games."""
    y = np.asarray(labels)
    p = np.asarray(proba, dtype=float)
    if len(y) < 20 or len(np.unique(y)) < 2:
        return 1.0
    scores = [(log_loss(y, calibrate_proba(p, t)), t) for t in grid]
    return float(min(scores)[1])


def select_C(
    X: pd.DataFrame,
    y: pd.Series,
    year: np.ndarray,
    through_year: int,
    grid: tuple[float, ...] = C_GRID,
) -> float:
    """Inner walk-forward C search using only seasons before `through_year`."""
    labels: list[int] = []
    probs: dict[float, list[float]] = {float(c): [] for c in grid}
    for inner_year in range(INNER_WALK_FORWARD_FROM, through_year):
        train_mask, test_mask = year < inner_year, year == inner_year
        if not train_mask.any() or not test_mask.any():
            continue
        labels.extend(y[test_mask].tolist())
        for c in grid:
            model = make_pipeline(float(c)).fit(X[train_mask], y[train_mask])
            probs[float(c)].extend(model.predict_proba(X[test_mask])[:, 1].tolist())

    if not labels:
        return 1.0
    scores = {
        c: log_loss(labels, predictions)
        for c, predictions in probs.items()
        if predictions
    }
    return float(min(scores, key=scores.get))


def feature_importance(
    model: Pipeline, feature_names: list[str], limit: int | None = 25
) -> list[dict]:
    """Rank standardized logistic-regression coefficients by absolute size."""
    coefs = model.named_steps["clf"].coef_[0]
    ranked = sorted(zip(feature_names, coefs), key=lambda item: abs(item[1]), reverse=True)
    if limit is not None:
        ranked = ranked[:limit]
    return [
        {
            "feature": name,
            "coefficient": round(float(value), 4),
            "magnitude": round(abs(float(value)), 4),
        }
        for name, value in ranked
    ]


def train(
    seasons=DEFAULT_SEASONS,
    C: float | None = None,
    walk_forward_from: int = 2021,
    save: bool = True,
    tune: bool = True,
) -> dict:
    """Fit on all `seasons`, with nested time-ordered validation reported first."""
    seasons = tuple(seasons)
    ds = build_dataset(seasons)
    teams = sorted(set(ds["home_team"]) | set(ds["away_team"]))
    X = make_features(ds, teams)
    y = ds["home_win"].astype(int)
    year = ds["season"].to_numpy()

    walk_forward = []
    raw_y: list[int] = []
    raw_p: list[float] = []
    cal_y: list[int] = []
    cal_p: list[float] = []
    oof_index: list[int] = []
    for test_year in range(walk_forward_from, max(seasons) + 1):
        train_mask, test_mask = year < test_year, year == test_year
        if not train_mask.any() or not test_mask.any():
            continue
        year_C = (
            select_C(X, y, year, through_year=test_year)
            if tune and C is None else float(C if C is not None else 1.0)
        )
        season_model = make_pipeline(year_C).fit(X[train_mask], y[train_mask])
        year_y = y[test_mask]
        year_raw = season_model.predict_proba(X[test_mask])[:, 1]

        # Fit on OOF predictions strictly before this season, then score it.
        year_temperature = fit_temperature(raw_y, raw_p) if raw_y else 1.0
        year_cal = calibrate_proba(year_raw, year_temperature)
        walk_forward.append({
            "season": test_year,
            "games": int(test_mask.sum()),
            "selected_C": year_C,
            "temperature": year_temperature,
            **score_predictions(year_y, year_cal),
            "raw_log_loss": score_predictions(year_y, year_raw)["log_loss"],
            "raw_brier": score_predictions(year_y, year_raw)["brier"],
        })
        raw_y.extend(year_y.tolist())
        raw_p.extend(year_raw.tolist())
        cal_y.extend(year_y.tolist())
        cal_p.extend(year_cal.tolist())
        oof_index.extend(np.where(test_mask)[0].tolist())

    # Reliability curve from calibrated out-of-fold predictions. Fixed 0-10% ..
    # 90-100% bins keep the UI axis stable across retrains.
    oof_probs = np.asarray(cal_p)
    oof_labels = np.asarray(cal_y)
    edges = np.linspace(0.0, 1.0, 11)
    calibration = []
    for i in range(10):
        upper_inclusive = oof_probs <= edges[i + 1] if i == 9 else oof_probs < edges[i + 1]
        mask = (oof_probs >= edges[i]) & upper_inclusive
        count = int(mask.sum())
        mean_pred = float(oof_probs[mask].mean()) if count else float((edges[i] + edges[i + 1]) / 2)
        actual = float(oof_labels[mask].mean()) if count else mean_pred
        calibration.append({
            "bin": f"{int(edges[i] * 100)}-{int(edges[i + 1] * 100)}%",
            "mean_pred": mean_pred,
            "actual": actual,
            "count": count,
        })

    final_C = (
        select_C(X, y, year, through_year=max(seasons) + 1)
        if tune and C is None else float(C if C is not None else 1.0)
    )
    final_model = make_pipeline(final_C).fit(X, y)
    final_temperature = fit_temperature(raw_y, raw_p) if raw_y else 1.0
    in_sample = score_predictions(
        y, calibrate_proba(final_model.predict_proba(X)[:, 1], final_temperature)
    )
    all_importance = feature_importance(
        final_model, list(X.columns), limit=len(X.columns)
    )
    importance = all_importance[:25]

    # Benchmarks use the same out-of-fold games the model was scored on.
    idx = np.asarray(oof_index, dtype=int)
    oof_y = y.iloc[idx]
    spread = ds["bet_home_spread"].iloc[idx]
    market_mask = spread.notna()
    full_spread = ds["bet_home_spread"]
    full_market_mask = full_spread.notna()
    baselines = {
        "always_home": float(oof_y.mean()),
        "market_favorite": float(
            ((spread[market_mask] > 0).astype(int) == oof_y[market_mask]).mean()
        ),
        "market_games": int(market_mask.sum()),
        "always_home_all": float(y.mean()),
        "market_favorite_all": float(
            ((full_spread[full_market_mask] > 0).astype(int)
             == y[full_market_mask]).mean()
        ),
    }

    artifact = {
        "model": final_model,
        "teams": teams,
        "seasons": list(seasons),
        "rolling_window": ROLLING_WINDOW,
        "weekly_window": WEEKLY_WINDOW,
        "C": final_C,
        "temperature": final_temperature,
        "tuned": bool(tune and C is None),
        "c_grid": list(C_GRID),
        "temperature_grid": list(T_GRID),
        "in_sample": in_sample,
        "walk_forward": walk_forward,
        "calibration": calibration,
        "baselines": baselines,
        "training_games": int(len(ds)),
        "feature_count": len(NUMERIC_FEATURES) + 2 * len(teams),
        "numeric_feature_count": len(NUMERIC_FEATURES),
        "num_teams": len(teams),
        "player_stats": player_stats_coverage(),
        "player_feature_count": len(PLAYER_COLUMNS) * 3,
        "weekly_player_feature_count": len(WEEKLY_MODEL_COLUMNS) * 3,
        "weekly_player_available_count": len(WEEKLY_PLAYER_COLUMNS) * 3,
        "weekly_player_source": "nflreadpy weekly player stats (current season, prior games)",
        "feature_importance": importance,
        "player_feature_importance": [
            item for item in all_importance if "_player_" in item["feature"]
        ],
        "weekly_feature_importance": [
            item for item in all_importance if "_weekly_" in item["feature"]
        ],
    }

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, MODEL_PATH)
        METRICS_PATH.write_text(json.dumps({
            key: value for key, value in artifact.items() if key != "model"
        }, indent=2), encoding="utf-8")

    return artifact


def predict_games(artifact: dict, games: pd.DataFrame) -> pd.DataFrame:
    """Return temperature-calibrated home-win probabilities for processed games."""
    X = make_features(games, artifact["teams"])
    raw = artifact["model"].predict_proba(X)[:, 1]
    proba = calibrate_proba(raw, artifact.get("temperature", 1.0))
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
    parser.add_argument(
        "--C", type=float, default=None,
        help="fixed L2 strength (default: nested walk-forward tuning)",
    )
    parser.add_argument(
        "--no-tune", action="store_true",
        help="skip the C search and use C=1.0",
    )
    parser.add_argument("--no-save", action="store_true", help="fit without writing artifacts")
    args = parser.parse_args()

    artifact = train(
        C=args.C, save=not args.no_save, tune=not args.no_tune and args.C is None
    )

    print(f"Trained on {artifact['training_games']:,} games "
          f"({min(artifact['seasons'])}-{max(artifact['seasons'])})")
    print(f"Features: {artifact['numeric_feature_count']} numeric + "
          f"{2 * artifact['num_teams']} team indicators")
    tuning = "nested walk-forward" if artifact["tuned"] else f"fixed C={artifact['C']}"
    print(f"Tuning: {tuning} on C={list(C_GRID)} -> C={artifact['C']}")
    print(f"Calibration: temperature grid {list(T_GRID)} -> T={artifact['temperature']:.3f}")
    print(f"Player stats: {artifact['player_stats']['dataset']} "
          f"({artifact['player_stats']['seasons'][0]}-"
          f"{artifact['player_stats']['seasons'][1]}, "
          f"{artifact['player_feature_count']} prior-season inputs)")
    print(f"Weekly form: nflreadpy, {artifact['weekly_window']}-game window, "
          f"{artifact['weekly_player_feature_count']} current-season inputs")

    print("\nWalk-forward validation (train on prior seasons):")
    print(f"  {'Season':<8}{'Games':>7}{'Acc':>8}{'AUC':>8}{'LogLoss':>10}{'Brier':>8}{'C':>7}{'Temp':>7}")
    for row in artifact["walk_forward"]:
        print(f"  {row['season']:<8}{row['games']:>7}{row['accuracy']:>8.3f}"
              f"{row['roc_auc']:>8.3f}{row['log_loss']:>10.3f}{row['brier']:>8.3f}"
              f"{row['selected_C']:>7.2f}{row['temperature']:>7.2f}")

    recent = artifact["walk_forward"][-3:]
    if recent:
        print(f"\nLast {len(recent)} seasons avg: "
              f"accuracy={np.mean([r['accuracy'] for r in recent]):.3f}, "
              f"AUC={np.mean([r['roc_auc'] for r in recent]):.3f}, "
              f"Brier={np.mean([r['brier'] for r in recent]):.3f} "
              f"(raw Brier={np.mean([r['raw_brier'] for r in recent]):.3f})")
    print(f"Baselines on the same OOF games -> always home: "
          f"{artifact['baselines']['always_home']:.3f}, "
          f"market favorite: {artifact['baselines']['market_favorite']:.3f} "
          f"({artifact['baselines']['market_games']:,} games)")
    print(f"In-sample (final 10-season fit): accuracy="
          f"{artifact['in_sample']['accuracy']:.3f}, "
          f"AUC={artifact['in_sample']['roc_auc']:.3f}")
    if not args.no_save:
        print(f"\nSaved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
