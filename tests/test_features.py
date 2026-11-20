"""Leakage guards for game-level feature engineering.

All loaders are replaced with in-memory fixtures, so these tests never touch
nflreadpy, Kaggle, or the network.
"""

import numpy as np
import pandas as pd
import pytest

from src import features


def _games() -> pd.DataFrame:
    """Four games; each team plays four times with strictly increasing dates."""
    rows = []
    for i in range(1, 5):
        rows.append({
            "game_id": f"g{i}",
            "gameday": f"2025-09-{i * 7:02d}",
            "home_team": "AAA" if i % 2 == 1 else "BBB",
            "away_team": "BBB" if i % 2 == 1 else "AAA",
        })
    games = pd.DataFrame(rows)
    for side in ("home", "away"):
        for col in features.ROLLING_BASE_COLUMNS:
            values = np.arange(1, len(games) + 1, dtype=float)
            games[f"{side}_{col}"] = values + (0.0 if side == "home" else 0.5)
    return games


def test_rolling_form_ignores_current_game_stats():
    games = _games()
    baseline = features.add_rolling_form(games, window=2)

    # Only the last game is altered. It is the final game for both teams, so no
    # later row can legitimately depend on it either.
    leaked = games.copy()
    last = leaked.index[-1]
    for side in ("home", "away"):
        for col in features.ROLLING_BASE_COLUMNS:
            leaked.at[last, f"{side}_{col}"] = 9999.0
    after = features.add_rolling_form(leaked, window=2)

    rolling_columns = [
        col for col in baseline.columns
        if col.startswith(("home_prev", "away_prev"))
    ]
    assert rolling_columns
    pd.testing.assert_frame_equal(
        baseline[rolling_columns].reset_index(drop=True),
        after[rolling_columns].reset_index(drop=True),
    )


def test_rolling_form_first_game_has_no_history():
    out = features.add_rolling_form(_games(), window=3)
    for side, team in (("home", "AAA"), ("away", "BBB")):
        first = out[out[f"{side}_team"] == team].sort_values("gameday").iloc[0]
        assert pd.isna(first[f"{side}_prev3_off_epa"])


def test_rolling_window_only_uses_the_last_n_games():
    out = features.add_rolling_form(_games(), window=2)
    third = out[out["game_id"] == "g3"].iloc[0]
    # AAA played g1 at home (1.0) and g2 away (2.5) before g3.
    assert third["home_prev2_off_epa"] == pytest.approx((1.0 + 2.5) / 2)


def test_adjusted_epa_is_built_from_prior_game_rates():
    out = features.add_rolling_form(_games(), window=3)
    for side, opponent in (("home", "away"), ("away", "home")):
        expected_off = (
            out[f"{side}_prev3_off_epa_per_play"]
            + out[f"{opponent}_prev3_def_epa_per_play"]
        )
        expected_def = (
            out[f"{side}_prev3_def_epa_per_play"]
            + out[f"{opponent}_prev3_off_epa_per_play"]
        )
        pd.testing.assert_series_equal(
            out[f"{side}_prev3_adj_off_epa"], expected_off, check_names=False
        )
        pd.testing.assert_series_equal(
            out[f"{side}_prev3_adj_def_epa"], expected_def, check_names=False
        )


def test_weekly_player_form_carries_latest_prior_week(monkeypatch):
    def fake_form(seasons, window=4):
        rows = [{
            "season": 2026,
            "week": week,
            "team": "AAA",
            "qb_name": f"QB{week}",
            "rb_name": f"RB{week}",
            "wr_name": f"WR{week}",
            "qb_ypa": 10.0 + week,
            "qb_epa_att": 0.1 * week,
            "rb_ypc": 4.0 + week,
            "wr_ypr": 12.0 + week,
            "top3_rec_yards": 100.0 * week,
        } for week in (1, 2, 3)]
        return pd.DataFrame(rows)

    monkeypatch.setattr(features, "build_weekly_team_form", fake_form)
    games = pd.DataFrame([
        {"season": 2026, "week": 2, "home_team": "AAA", "away_team": "AAA"},
        {"season": 2026, "week": 4, "home_team": "AAA", "away_team": "AAA"},
    ])

    out = features.add_weekly_player_features(games)
    # A played week uses its own row; an unplayed week carries week 3 forward.
    assert out.loc[0, "home_weekly_qb_name"] == "QB2"
    assert out.loc[1, "home_weekly_qb_name"] == "QB3"
    assert out.loc[1, "home_weekly_qb_ypa"] == 13.0
    assert out.loc[1, "home_weekly_top3_rec_yards"] == 300.0


def test_prior_season_player_features_never_use_current_season(monkeypatch):
    stats = pd.DataFrame([
        {
            "season": season, "team": "AAA",
            **{col: float(season % 100) for col in features.PLAYER_COLUMNS},
            **{col: label for col in features.PLAYER_NAME_COLUMNS},
        }
        for season, label in ((2024, "old"), (2025, "current"))
    ])
    monkeypatch.setattr(features, "build_team_season_player_stats", lambda: stats)
    games = pd.DataFrame([
        {"season": 2025, "home_team": "AAA", "away_team": "AAA"},
        {"season": 2026, "home_team": "AAA", "away_team": "AAA"},
    ])

    out = features.add_player_features(games)
    assert out.loc[0, "home_player_qb_rating"] == 24.0
    assert out.loc[0, "home_player_qb_name"] == "old"
    assert out.loc[1, "home_player_qb_rating"] == 25.0
    assert out.loc[1, "home_player_qb_name"] == "current"
