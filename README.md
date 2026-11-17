# NFL Weekly Game Predictor

Predict the outcomes of weekly NFL games using historical play-by-play and
schedule data from the [nflverse](https://nflverse.nflverse.com/) via
`nflreadpy`, enriched with historical scores and betting lines from the Kaggle
[NFL Scores & Betting Data](https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data)
dataset.

## Tech stack

| Tool | Purpose |
| --- | --- |
| Python | Language / runtime |
| nflreadpy | Load nflverse data (schedules, stats, play-by-play, etc.) |
| pandas | Data cleaning, feature engineering, analysis |
| scikit-learn | Machine learning models and evaluation |
| seaborn | Data visualization / EDA |
| pyarrow | Bridge for converting nflreadpy's Polars output to pandas |
| kagglehub | Download the Kaggle scores + betting-lines dataset |

> Note: `nflreadpy` returns **Polars** DataFrames. The helpers in `src/data.py`
> convert them to pandas with `.to_pandas()` (which requires `pyarrow`).

## Project structure

```
nfl-game-predictor/
├── data/
│   ├── raw/          # Kaggle CSVs (spreadspoke_scores.csv, nfl_teams.csv)
│   └── processed/    # cleaned datasets and engineered features
├── notebooks/        # exploratory analysis / experiments
├── src/
│   ├── data.py       # nflreadpy -> pandas loaders
│   ├── betting_data.py  # Kaggle scores + betting-line loaders
│   ├── features.py   # EPA / pace / rest / bye + rolling form
│   └── ...           # models
├── tests/            # unit tests
├── requirements.txt
└── README.md
```

## Setup

```powershell
# create + activate virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt
```

(If PowerShell blocks the activation script, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.)

## Data sources

- **nflverse (nflreadpy)** — live schedules, team/player stats. See `src/data.py`.
- **Kaggle scores + betting lines** — 14,371 games from 1966–2025 with closing
  spreads, over/under lines, stadium and weather. The CSVs are cached in
  `data/raw/`. If missing, `src/betting_data.py` re-downloads the latest version
  automatically:

```python
from src.betting_data import load_scores, load_teams

games = load_scores()  # adds home_id, away_id, home_win, home_margin,
                       # total_points, home_spread
teams = load_teams()   # team name -> abbreviation mapping
```

## Feature availability

| Feature | Source | Implementation |
| --- | --- | --- |
| Offensive EPA | nflreadpy PBP | `epa` summed/averaged where `posteam` = offense |
| Defensive EPA | nflreadpy PBP | `epa` where `defteam` = defense (EPA allowed; higher = worse) |
| Pace | nflreadpy PBP | offensive plays per game |
| Rest days | nflreadpy schedules | `home_rest` / `away_rest` (days since previous game) |
| Bye week | derived | `rest >= 12` days as a proxy |
| Spread / over-under / weather | Kaggle | joined on season + date + teams (2024: 282/285 games) |

The Kaggle data does **not** contain EPA or pace; nflreadpy covers those. Kaggle
adds betting lines and historical context. Team IDs `LAR`/`LVR` in Kaggle are
mapped to nflverse's `LA`/`LV`.

Build the game-level table (use `add_rolling_form` for leakage-free prediction
features — it only uses games played before the target game):

```python
from src.features import build_game_features, add_rolling_form

games = build_game_features(seasons=range(2018, 2026))
model_input = add_rolling_form(games, window=3)
```

## Status

Data loading and feature engineering are ready. Modeling code is next.
