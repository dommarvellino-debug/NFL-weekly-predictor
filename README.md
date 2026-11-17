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
├── models/           # trained model artifact + metrics JSON
├── src/
│   ├── data.py       # nflreadpy -> pandas loaders
│   ├── betting_data.py  # Kaggle scores + betting-line loaders
│   ├── features.py   # EPA / pace / rest / bye + rolling form
│   └── model.py      # logistic regression training + prediction
├── tests/            # unit tests
├── web/              # static GridironML-style UI (4 screens)
│   ├── index.html          # Weekly Matchup Board & AI Picks (landing)
│   ├── pick-simulator.html # Solo Pick Simulator & AI Comparison
│   ├── model-metrics.html  # Model Performance & Calibration
│   ├── game-insights.html  # Game Insights & ML Deep Dive
│   ├── theme.js            # shared Tailwind config / design tokens
│   ├── styles.css          # shared base styles
│   └── logo.svg            # local logo (no external image dependency)
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

## Static web UI

The four-screen GridironML UI lives in `web/`. It is plain HTML/CSS/JS with a
shared Tailwind config, so no build step or backend is required. Tailwind and
Google Fonts are loaded from CDNs; the logo is local.

```powershell
cd web
python -m http.server 8000
```

Then open <http://127.0.0.1:8000/> (Weekly Board). The header nav links to all
four screens. You can also open an individual `*.html` file directly in a
browser, but serving over HTTP is recommended so shared assets resolve
consistently.

> The screens currently display realistic mock Week 10 data and client-side
> interactions (filters, pick selection, scenario slider). They are a UI shell
> ready to be wired to real model predictions once modeling is in place.

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

## Logistic-regression model

`src/model.py` trains an L2 logistic regression to predict P(home win) on the
ten most recent complete seasons (**2016-2025**, 2,751 decided games; ties are
dropped). Features are pre-game only:

- rolling (previous 3 games) offensive EPA, defensive EPA and pace for each
  team, plus home-minus-away differences
- rest days and the bye proxy for each side, plus their differences
- one-hot team identity (lets the model learn team strength)

Same-game EPA/pace is excluded to avoid leaking the result. The Kaggle closing
spread is held out of the model and used only as a market baseline.

```powershell
python -m src.model
```

This prints time-ordered validation and saves `models/logistic_regression.joblib`
plus `models/logistic_regression.json` (metrics). Validation is walk-forward —
each season is scored by a model trained only on earlier seasons:

| Test season | Accuracy | ROC AUC | Brier |
| --- | --- | --- | --- |
| 2021 | 0.658 | 0.651 | 0.239 |
| 2022 | 0.550 | 0.582 | 0.251 |
| 2023 | 0.575 | 0.604 | 0.244 |
| 2024 | 0.621 | 0.680 | 0.224 |
| 2025 | 0.634 | 0.674 | 0.227 |

Last-three-season average: **0.610 accuracy / 0.652 AUC / 0.232 Brier**, versus
baselines of 0.550 for always picking the home team and 0.667 for the closing
market favorite. The model beats the naive baseline but not the betting market,
which is a useful reference point rather than a surprise.

Use the fitted model:

```python
from src.model import load_model, predict_games
from src.features import build_game_features, add_rolling_form

artifact = load_model()
games = add_rolling_form(build_game_features(2026))
picks = predict_games(artifact, games)   # game_id, teams, home_win_prob
```

## Status

Data loading, feature engineering, the logistic-regression model, and the
four-screen static UI are ready. Next: wire the model's predictions into `web/`.
