"""Shared test setup: keep every test offline and deterministic."""

import pytest

from src import weekly_players


@pytest.fixture(autouse=True)
def clear_weekly_form_cache():
    """The weekly loader is lru_cached; each test injects its own fake rows."""
    weekly_players._build_cached.cache_clear()
    yield
    weekly_players._build_cached.cache_clear()
