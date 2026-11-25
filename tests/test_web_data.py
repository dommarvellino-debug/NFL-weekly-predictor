"""Contracts for the JSON payloads served to the web UI.

These checks run offline against the committed exports in ``web/data``. They
guard against schema drift, impossible probabilities, mismatched matchups,
fabricated betting lines, and stale-but-future prediction dates.
"""

from __future__ import annotations

import json
import statistics
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parents[1] / "web" / "data"
CONFIDENCE_LABELS = {"high", "moderate", "low"}
MAX_PREDICTION_AGE_DAYS = 14
FORBIDDEN_LINE_KEYS = {
    "spread", "closing_spread", "over_under", "total", "moneyline",
    "market_prob", "market_win_prob", "betting_line",
}


@pytest.fixture(scope="module")
def predictions() -> dict:
    return json.loads((DATA_DIR / "predictions.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def metrics() -> dict:
    return json.loads((DATA_DIR / "model-metrics.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def games(predictions) -> list[dict]:
    return predictions["games"]


def test_predictions_top_level_schema(predictions):
    assert set(predictions) == {"season", "week", "model", "note", "games"}
    assert isinstance(predictions["season"], int)
    assert 2016 <= predictions["season"] <= date.today().year + 1
    assert isinstance(predictions["week"], int)
    assert 1 <= predictions["week"] <= 18
    assert isinstance(predictions["model"], dict)
    assert isinstance(predictions["games"], list)
    assert predictions["games"]


def test_game_schema_and_unique_ids(games):
    required = {
        "game_id", "gameday", "weekday", "gametime", "stadium",
        "home", "away", "predicted_winner", "confidence", "edge_pp",
    }
    side_required = {"abbr", "name", "record", "win_prob", "features", "players"}
    assert len({game["game_id"] for game in games}) == len(games)
    for game in games:
        assert required <= game.keys()
        for side in ("home", "away"):
            assert side_required <= game[side].keys()
            assert game[side]["abbr"]
            assert game[side]["name"]


def test_game_id_matches_matchup_and_week(games, predictions):
    for game in games:
        parts = game["game_id"].split("_")
        assert len(parts) == 4, game["game_id"]
        season, week, away_abbr, home_abbr = parts
        assert (int(season), int(week)) == (
            predictions["season"], predictions["week"]
        )
        assert away_abbr == game["away"]["abbr"]
        assert home_abbr == game["home"]["abbr"]


def test_probabilities_are_valid_and_consistent(games):
    for game in games:
        p_home = game["home"]["win_prob"]
        p_away = game["away"]["win_prob"]
        assert 0.0 <= p_home <= 1.0
        assert 0.0 <= p_away <= 1.0
        assert p_home + p_away == pytest.approx(1.0, abs=1e-3)

        expected_winner = (
            game["home"]["abbr"] if p_home >= p_away else game["away"]["abbr"]
        )
        assert game["predicted_winner"] == expected_winner
        assert game["confidence"] in CONFIDENCE_LABELS
        assert game["edge_pp"] == pytest.approx(
            round(abs(p_home - 0.5) * 100, 1), abs=0.1
        )


def test_predictions_contain_no_betting_lines(games):
    """2026 lines are unavailable; the UI must never show an invented market line."""
    for game in games:
        for mapping in (game, game["home"], game["away"]):
            assert not FORBIDDEN_LINE_KEYS.intersection(
                key.lower() for key in mapping
            )


def test_gamedays_form_one_valid_slate(games, predictions):
    today = date.today()
    slate = []
    for game in games:
        gameday = date.fromisoformat(game["gameday"])
        assert gameday.year == predictions["season"]
        assert gameday <= today + timedelta(days=1)  # no future-dated picks
        assert game["weekday"] == gameday.strftime("%A")
        datetime.strptime(game["gametime"], "%H:%M")
        slate.append(gameday)
    assert max(slate) - min(slate) <= timedelta(days=7)


def test_predictions_are_fresh_during_the_season(games):
    today = date.today()
    in_season = today.month in (9, 10, 11, 12) or (
        today.month == 1 and today.day <= 15
    )
    if not in_season:
        pytest.skip("NFL offseason; there is no weekly slate to keep fresh")
    latest = max(date.fromisoformat(game["gameday"]) for game in games)
    age = (today - latest).days
    assert age <= MAX_PREDICTION_AGE_DAYS, (
        f"web/data/predictions.json is {age} days old (latest {latest}); "
        "rerun python -m src.predict"
    )


def test_model_metrics_schema(metrics):
    required = {
        "seasons", "training_games", "features", "num_teams", "numeric_features",
        "C", "temperature", "tuned", "c_grid", "temperature_grid",
        "rolling_window", "in_sample", "walk_forward", "calibration",
        "baselines", "player_stats", "player_feature_count", "weekly_window",
        "weekly_player_feature_count", "weekly_player_available_count",
        "weekly_player_source", "feature_importance",
        "player_feature_importance", "weekly_feature_importance",
        "recent_average",
    }
    assert required <= metrics.keys()
    assert metrics["tuned"] is True
    assert metrics["C"] in metrics["c_grid"]
    assert metrics["temperature"] in metrics["temperature_grid"]
    assert metrics["features"] == (
        metrics["numeric_features"] + 2 * metrics["num_teams"]
    )


def test_metric_values_are_bounded(metrics):
    for block in ("in_sample", "recent_average"):
        for name in ("accuracy", "roc_auc", "brier"):
            assert 0.0 <= metrics[block][name] <= 1.0, (block, name)
    assert metrics["in_sample"]["log_loss"] >= 0.0

    assert metrics["walk_forward"]
    for season in metrics["walk_forward"]:
        assert season["games"] > 0
        assert season["selected_C"] in metrics["c_grid"]
        for name in ("accuracy", "roc_auc", "brier"):
            assert 0.0 <= season[name] <= 1.0
        for name in ("log_loss", "raw_log_loss", "raw_brier"):
            assert season[name] >= 0.0

    for name in ("always_home", "market_favorite", "always_home_all",
                 "market_favorite_all"):
        assert 0.0 <= metrics["baselines"][name] <= 1.0
    assert metrics["baselines"]["market_games"] > 0


def test_calibration_counts_match_walk_forward(metrics):
    assert sum(bucket["count"] for bucket in metrics["calibration"]) == sum(
        season["games"] for season in metrics["walk_forward"]
    )
    for bucket in metrics["calibration"]:
        assert {"bin", "mean_pred", "actual", "count"} <= bucket.keys()
        assert bucket["count"] >= 0
        assert 0.0 <= bucket["mean_pred"] <= 1.0
        assert 0.0 <= bucket["actual"] <= 1.0


def test_recent_average_matches_last_three_seasons(metrics):
    last_three = metrics["walk_forward"][-3:]
    for name in ("accuracy", "roc_auc", "brier"):
        expected = round(statistics.mean(season[name] for season in last_three), 4)
        assert metrics["recent_average"][name] == pytest.approx(
            expected, abs=1e-4
        )


def test_feature_importance_is_ranked(metrics):
    importance = metrics["feature_importance"]
    assert importance
    magnitudes = [item["magnitude"] for item in importance]
    assert magnitudes == sorted(magnitudes, reverse=True)
    for item in importance:
        assert item["magnitude"] == pytest.approx(
            abs(item["coefficient"]), abs=1e-4
        )


def test_exports_agree_on_model_configuration(predictions, metrics):
    metadata = predictions["model"]
    assert metadata["features"] == metrics["features"]
    assert metadata["C"] == metrics["C"]
    assert metadata["temperature"] == metrics["temperature"]
    assert metadata["training_games"] == metrics["training_games"]
    assert metadata["seasons"] == metrics["seasons"]
    assert metadata["walk_forward_accuracy"] == pytest.approx(
        metrics["recent_average"]["accuracy"], abs=1e-4
    )
    assert metadata["walk_forward_auc"] == pytest.approx(
        metrics["recent_average"]["roc_auc"], abs=1e-4
    )
