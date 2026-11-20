"""Contracts for the model matrix and probability calibration."""

import numpy as np
import pandas as pd
import pytest

from src import features, model
from src.player_stats import PLAYER_COLUMNS


def test_numeric_feature_contract():
    assert len(model.NUMERIC_FEATURES) == 48
    assert len(set(model.NUMERIC_FEATURES)) == len(model.NUMERIC_FEATURES)
    # QB EPA/attempt is insight-only, never a model input.
    assert not any("qb_epa_att" in col for col in model.NUMERIC_FEATURES)
    assert len(model.WEEKLY_MODEL_COLUMNS) == 4


def test_make_features_has_stable_columns():
    df = pd.DataFrame({
        "home_team": ["AAA"],
        "away_team": ["BBB"],
        "home_rest": [7],
        "away_rest": [6],
        "home_had_bye": [0],
        "away_had_bye": [1],
    })
    for col in (*features.ROLLING_BASE_COLUMNS, *features.MATCHUP_COLUMNS):
        for side in ("home", "away"):
            df[f"{side}_prev3_{col}"] = [1.0]
    for col in PLAYER_COLUMNS:
        for side in ("home", "away"):
            df[f"{side}_player_{col}"] = [2.0]
    for col in model.WEEKLY_MODEL_COLUMNS:
        df[f"home_weekly_{col}"] = [3.0]
        df[f"away_weekly_{col}"] = [1.0]

    X = model.make_features(df, ["AAA", "BBB"])
    assert X.shape == (1, 52)
    assert list(X.columns) == list(model.NUMERIC_FEATURES) + [
        "home_AAA", "home_BBB", "away_AAA", "away_BBB",
    ]
    assert X.loc[0, "diff_weekly_qb_ypa"] == pytest.approx(2.0)
    assert X.loc[0, "bye_diff"] == pytest.approx(-1.0)


def test_calibrate_proba_temperature_one_is_identity():
    p = np.array([0.05, 0.2, 0.5, 0.8, 0.95])
    np.testing.assert_allclose(model.calibrate_proba(p, 1.0), p, atol=1e-12)


def test_calibrate_proba_preserves_ranking_and_bounds():
    p = np.array([0.1, 0.4, 0.6, 0.9])
    out = model.calibrate_proba(p, 1.5)
    assert np.all((out > 0.0) & (out < 1.0))
    assert np.all(np.diff(out) > 0)


def test_fit_temperature_needs_enough_out_of_fold_games():
    assert model.fit_temperature([0, 1] * 5, [0.4, 0.6] * 5) == 1.0
