"""Leakage guards for current-season weekly player form."""

import pandas as pd
import pytest

from src import weekly_players


def _row(week: int, player_id: str, name: str, position: str, **stats) -> dict:
    row = {
        "season": 2025,
        "week": week,
        "season_type": "REG",
        "game_id": f"2025_{week:02d}_AAA_BBB",
        "team": "AAA",
        "player_id": player_id,
        "player_name": name,
        "position": position,
        "attempts": 0,
        "completions": 0,
        "passing_yards": 0,
        "passing_epa": 0.0,
        "carries": 0,
        "rushing_yards": 0,
        "receptions": 0,
        "targets": 0,
        "receiving_yards": 0,
    }
    row.update(stats)
    return row


def _player_rows() -> list[dict]:
    """Three weeks of one QB/RB/WR; the QB's week 3 is a huge outlier."""
    rows = []
    for week in (1, 2, 3):
        rows.append(_row(
            week, "qb1", "Q.One", "QB",
            attempts=10, completions=7,
            passing_yards=100 if week < 3 else 1000,
            passing_epa=1.0,
        ))
        rows.append(_row(
            week, "rb1", "R.One", "RB",
            carries=5, rushing_yards=50,
        ))
        rows.append(_row(
            week, "wr1", "W.One", "WR",
            receptions=5, targets=7, receiving_yards=50,
        ))
    return rows


def test_weekly_form_excludes_the_current_game(monkeypatch):
    monkeypatch.setattr(
        weekly_players.nfl_data, "load_player_stats", lambda seasons: pd.DataFrame(_player_rows())
    )
    form = weekly_players.build_weekly_team_form(2025, window=4).set_index("week")

    # Week 1 has no prior games, so the loader emits no row at all.
    assert 1 not in form.index
    assert form.loc[2, "qb_ypa"] == pytest.approx(10.0)
    # Week 3's 1,000 yards must not appear: week 3 uses only weeks 1-2.
    assert form.loc[3, "qb_ypa"] == pytest.approx(10.0)
    assert form.loc[3, "wr_ypr"] == pytest.approx(10.0)
    assert form.loc[3, "qb_name"] == "Q.One"


def test_weekly_form_does_not_carry_stats_across_seasons(monkeypatch):
    prior = _row(18, "qb1", "Q.One", "QB", attempts=10, passing_yards=1000)
    prior["season"] = 2024
    prior["game_id"] = "2024_18_AAA_BBB"
    monkeypatch.setattr(
        weekly_players.nfl_data, "load_player_stats",
        lambda seasons: pd.DataFrame([prior] + _player_rows()),
    )

    form = weekly_players.build_weekly_team_form((2024, 2025), window=4)
    # 2025 week 2 must use only 2025 week 1 (10.0 YPA), not the huge 2024 week 18.
    week_two = form[(form["season"] == 2025) & (form["week"] == 2)].iloc[0]
    assert week_two["qb_ypa"] == pytest.approx(10.0)


def test_weekly_form_rejects_invalid_window():
    with pytest.raises(ValueError, match="window"):
        weekly_players.build_weekly_team_form(2025, window=0)
