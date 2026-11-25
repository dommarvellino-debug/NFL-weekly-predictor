"""Offline tests for the nflverse loaders."""

import pandas as pd

from src import data


class _FakePolars:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def to_pandas(self) -> pd.DataFrame:
        return self.frame


def test_load_schedules_prefers_parquet(monkeypatch):
    expected = pd.DataFrame({"game_id": ["2026_01_A_B"], "season": [2026]})

    def fake_load(seasons):
        assert seasons == 2026
        return _FakePolars(expected)

    def no_csv(*args, **kwargs):
        raise AssertionError("CSV fallback should not be used when parquet works")

    monkeypatch.setattr(data.nfl, "load_schedules", fake_load)
    monkeypatch.setattr(pd, "read_csv", no_csv)

    pd.testing.assert_frame_equal(data.load_schedules(2026), expected)


def test_load_schedules_falls_back_to_csv_and_filters_seasons(monkeypatch):
    def missing_parquet(seasons):
        raise ConnectionError("games.parquet: 404 Not Found")

    csv = pd.DataFrame({
        "season": [2025, 2026, 2026, 2027],
        "week": [18, 11, 12, 1],
        "game_id": ["2025_18_A_B", "2026_11_C_D", "2026_12_E_F", "2027_01_G_H"],
    })

    monkeypatch.setattr(data.nfl, "load_schedules", missing_parquet)
    monkeypatch.setattr(pd, "read_csv", lambda url, **kwargs: csv)

    out = data.load_schedules([2026])

    assert out["game_id"].tolist() == ["2026_11_C_D", "2026_12_E_F"]
    assert out.index.tolist() == [0, 1]


def test_season_list_accepts_single_season_and_iterables():
    assert data._season_list(2026) == [2026]
    assert data._season_list((2025, 2026)) == [2025, 2026]
    assert data._season_list(range(2025, 2027)) == [2025, 2026]
