# NFL Weekly Game Predictor

[![tests](https://github.com/dommarvellino-debug/NFL-weekly-predictor/actions/workflows/tests.yml/badge.svg)](https://github.com/dommarvellino-debug/NFL-weekly-predictor/actions/workflows/tests.yml)

Predict the outcomes of weekly NFL games using historical play-by-play and
schedule data from the [nflverse](https://nflverse.nflverse.com/) via
`nflreadpy`, enriched with historical scores and betting lines from the Kaggle
[NFL Scores & Betting Data](https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data)
dataset, plus prior-season team player leaders from the Kaggle
[NFL Stats 1999-2022](https://www.kaggle.com/datasets/philiphyde1/nfl-stats-1999-2022)
dataset and current-season weekly player form from nflreadpy.

## Tech stack

| Tool | Purpose |
| --- | --- |
| Python | Language / runtime |
| nflreadpy | Load nflverse data (schedules, stats, play-by-play, etc.) |
| pandas | Data cleaning, feature engineering, analysis |
| scikit-learn | Machine learning models and evaluation |
| seaborn | Data visualization / EDA |
| pyarrow | Bridge for converting nflreadpy's Polars output to pandas |
| kagglehub | Download both Kaggle datasets (scores/betting lines and player stats) |

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
│   ├── player_stats.py  # Kaggle prior-season player leaders
│   ├── weekly_players.py # nflreadpy current-season prior-week player form
│   ├── features.py   # EPA rates / rest / bye / player joins + rolling form
│   ├── model.py      # tuned logistic regression + walk-forward evaluation
│   └── predict.py    # current-week predictions + web data exports
├── tests/            # offline leakage-guard tests (pytest)
├── pytest.ini         # pytest discovery / import path
├── web/              # static GridironML-style UI (4 screens)
│   ├── index.html          # Weekly Matchup Board & AI Picks (landing)
│   ├── pick-simulator.html # Solo Pick Simulator & AI Comparison
│   ├── model-metrics.html  # Model Performance & Calibration
│   ├── game-insights.html  # Game Insights & ML Deep Dive
│   ├── app.js              # shared JSON loader + all four screen renderers
│   ├── theme.js            # shared Tailwind config / design tokens
│   ├── styles.css          # shared base styles
│   ├── logo.svg            # local logo (no external image dependency)
│   └── data/
│       ├── predictions.json    # current week's model output
│       └── model-metrics.json  # validation + calibration for the UI
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

## Tests

The suite in `tests/` guards the pre-game-only contract with in-memory fixtures,
so it never downloads data or calls the network. It verifies that rolling team
form and weekly player form exclude the current game, that prior-season joins
never use the current season, that adjusted EPA is built only from prior games,
and that the model matrix and calibration keep their expected shape and ordering.

A second group of tests validates the JSON served to the UI: required keys,
probabilities that sum to one, game IDs that match the matchup and week, no
fabricated betting lines, one coherent slate of gamedays, fresh dates during
the season, and agreement between `predictions.json` and `model-metrics.json`.

```powershell
python -m pytest
```

The same command runs on every push and pull request via
`.github/workflows/tests.yml`, so the badge above turns red if either the
leakage guarantees or the exported data contract breaks.

## Static web UI

The four-screen GridironML UI lives in `web/`. It is plain HTML/CSS/JS with a
shared Tailwind config, so no build step or backend is required. Tailwind and
Google Fonts are loaded from CDNs; the logo is local.

All four screens are data-driven: `web/app.js` loads `web/data/predictions.json`
and `web/data/model-metrics.json` and renders the current slate, model inputs,
validation results, and calibration curve. There is no mock data in the UI.

```powershell
# regenerate the two JSON files after a new model/week (from repo root)
python -m src.predict

# serve the web folder
cd web
python -m http.server 8000
```

Then open <http://127.0.0.1:8000/> (Weekly Board). The header nav links to all
four screens:

- **Weekly Board** — featured lock, confidence filters, and every model pick.
- **Pick Simulator** — select your winners and compare them with the model.
- **Model Metrics** — nested walk-forward accuracy, Brier score, benchmarks,
  tuning/calibration details, prior-season and weekly player-feature impact, and
  a reliability diagram built from out-of-fold predictions.
- **Game Insights** — switch between games to inspect rolling EPA/play, success
  rate, opponent-adjusted EPA, pace, rest, bye, prior-season leaders, and
  current-season weekly player form behind each probability.

> Serve the folder over HTTP. Opening an `*.html` file with `file://` will block
> the `fetch()` calls that load the JSON data.

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

- **Kaggle player stats** — `philiphyde1/nfl-stats-1999-2022`. Despite the
  dataset name, the current version contains 2012–2025 seasons. The loader reads
  only the compact `yearly_player_stats_offense.csv` (not the 342 MB weekly
  table) and derives one row per team-season:

```python
import kagglehub

path = kagglehub.dataset_download("philiphyde1/nfl-stats-1999-2022")
print("Path to dataset files:", path)
```

```python
from src.player_stats import build_team_season_player_stats, player_stats_coverage

leaders = build_team_season_player_stats()  # QB/RB/receiver + receiving depth
coverage = player_stats_coverage()          # seasons, team-seasons, teams
```

- **nflreadpy weekly player stats** — current-season player-game rows. The
  loader computes each player's rolling sums over his previous games, then picks
  the team's leading QB/RB/receiver for each week. A merge-as-of carries the
  latest prior week forward, so an upcoming week can use the most recent
  completed week even before its own player rows exist.

```python
from src.weekly_players import build_weekly_team_form

form = build_weekly_team_form([2025, 2026], window=4)
# season, week, team, qb_name/rb_name/wr_name + rates and top-3 receiving yards
```

## Feature availability

| Feature | Source | Implementation |
| --- | --- | --- |
| Offensive EPA | nflreadpy PBP | `epa` summed and averaged per play where `posteam` = offense |
| Defensive EPA | nflreadpy PBP | `epa` where `defteam` = defense (EPA allowed; higher = worse) |
| Success rate | nflreadpy PBP | mean of nflverse `success` on offensive plays |
| Opponent-adjusted EPA | derived | own offensive EPA/play + opponent's defensive EPA/play allowed |
| Pace | nflreadpy PBP | offensive plays per game |
| Rest days | nflreadpy schedules | `home_rest` / `away_rest` (days since previous game) |
| Bye week | derived | `rest >= 12` days as a proxy |
| Prior-season player form | Kaggle player stats | leading QB rating/YPA, lead RB YPC, lead receiver YPR, top-3 receiving yards |
| Current-season weekly form | nflreadpy player stats | prior-game QB YPA, RB YPC, receiver YPR, top-3 receiving yards |
| Spread / over-under / weather | Kaggle | joined on season + date + teams (2024: 282/285 games) |

The scores/betting Kaggle data does **not** contain EPA or pace; nflreadpy covers
those. The Kaggle player dataset adds entering-season form: for a game in season
`S`, only leaders from season `S-1` are joined. The separate nflreadpy weekly
loader adds current-season form using only prior games (its rolling sums shift by
one game), so week 11 can use weeks 1-10 but never week 11 itself. Team IDs
`LAR`/`LVR` in Kaggle are mapped to nflverse's `LA`/`LV`.

Build the game-level table (use `add_rolling_form` for leakage-free prediction
features — it only uses games played before the target game):

```python
from src.features import build_game_features, add_rolling_form

games = build_game_features(seasons=range(2018, 2026), weekly_window=4)
model_input = add_rolling_form(games, window=3)
```

## Logistic-regression model

`src/model.py` trains an L2 logistic regression to predict P(home win) on the
ten most recent complete seasons (**2016-2025**, 2,751 decided games; ties are
dropped). Features are pre-game only:

- rolling (previous 3 games) offensive EPA, defensive EPA and pace for each
  team, plus home-minus-away differences
- opponent-adjusted EPA built from EPA per play: own offensive EPA/play plus the
  opponent's defensive EPA/play allowed, and own defensive EPA/play plus the
  opponent's offensive EPA/play (both sides plus differences)
- rest days and the bye proxy for each side, plus their differences
- prior-season QB rating/YPA, lead RB YPC, lead receiver YPR and top-3 receiving
  yards for each side, plus home-minus-away differences (15 inputs)
- current-season weekly QB YPA, lead RB YPC, lead receiver YPR and top-3
  receiving yards over a 4-game window (12 inputs; QB EPA/attempt is exported
  for insight but was too noisy to keep as a model input)
- one-hot team identity (lets the model learn team strength)

That is **116 features** in total (48 numeric + 68 team indicators). Same-game
EPA/pace and current-game player stats are excluded to avoid leaking the result.
EPA per play and success rate are computed and exported to the UI; an ablation
showed that the opponent-adjusted EPA columns were the useful form of the rate
signal, so raw per-play/success columns are not separate model inputs. The Kaggle
closing spread is held out of the model and used only as a market baseline.

```powershell
python -m src.model
```

This prints time-ordered validation and saves `models/logistic_regression.joblib`
plus `models/logistic_regression.json` (metrics). Validation is nested
walk-forward: each season is scored by a model trained only on earlier seasons,
and the L2 strength `C` is chosen by an inner walk-forward over even earlier
seasons. Probabilities are calibrated with a temperature fitted only on earlier
out-of-fold games.

| Test season | Accuracy | ROC AUC | Brier |
| --- | --- | --- | --- |
| 2021 | 0.620 | 0.635 | 0.244 |
| 2022 | 0.567 | 0.610 | 0.237 |
| 2023 | 0.582 | 0.612 | 0.238 |
| 2024 | 0.656 | 0.702 | 0.224 |
| 2025 | 0.613 | 0.665 | 0.230 |

Last-three-season average: **0.617 accuracy / 0.660 AUC / 0.230 Brier**, versus
baselines on the same out-of-fold games of 0.546 for always picking the home team
and 0.667 for the closing market favorite (1,382 games with a line). The final
model refit on all ten seasons scores 0.656 accuracy and 0.709 AUC in-sample;
that number is descriptive, not a validation estimate. The model beats the naive
baseline but not the betting market, which is a useful reference point rather
than a surprise. The top-3 feature upgrades produced a small improvement over the
previous 98-feature model (0.615 / 0.657 / 0.231), not a large one.

Training writes a calibration table from all 1,420 out-of-fold walk-forward
predictions, using fixed 0–10% through 90–100% probability bins. The final
temperature is stored in the artifact and applied by `predict_games`, so the UI
shows the same calibrated probabilities that were validated.

Training also exports standardized logistic coefficients (`feature_importance`),
the prior-season player subset (`player_feature_importance`), and the weekly-form
subset (`weekly_feature_importance`). The Model Metrics screen displays tuning,
calibration, both player-feature coverage blocks, and these coefficients; Game
Insights displays the actual leaders and their numbers for both teams.

Use the fitted model:

```python
from src.model import load_model, predict_games
from src.features import build_game_features, add_rolling_form

artifact = load_model()
games = add_rolling_form(build_game_features(2026))
picks = predict_games(artifact, games)   # game_id, teams, home_win_prob
```

## Weekly predictions

`src/predict.py` applies the fitted model to a 2026 game week. By default it
picks the first week with unplayed games; pass `--week N` for a specific week.
Team form is computed from completed 2025 + 2026 games, so every prediction
uses only information available before kickoff.

```powershell
python -m src.predict           # next upcoming week
python -m src.predict --week 12 # a specific week
```

This prints a readable pick table and writes two UI-facing files:
`web/data/predictions.json` and `web/data/model-metrics.json`. The predictions
file includes model metadata, kickoff/venue details, the pre-game inputs used
for each team, and the winner probability/confidence:

```json
{
  "season": 2026,
  "week": 11,
  "model": {
    "type": "logistic_regression",
    "training_games": 2751,
    "features": 116,
    "C": 0.03,
    "temperature": 1.5,
    "player_features": 15,
    "player_source": "philiphyde1/nfl-stats-1999-2022",
    "player_seasons": [2012, 2025],
    "weekly_player_features": 12,
    "weekly_player_window": 4,
    "weekly_player_source": "nflreadpy weekly player stats (current season, prior games)",
    "walk_forward_accuracy": 0.6171,
    "walk_forward_auc": 0.6596
  },
  "games": [
    {
      "game_id": "2026_11_MIA_BUF",
      "gameday": "2026-11-22",
      "weekday": "Sunday",
      "gametime": "13:00",
      "stadium": "Highmark Stadium",
      "home": {
        "abbr": "BUF",
        "name": "Buffalo Bills",
        "record": "8-1",
        "win_prob": 0.8121,
        "features": {
          "off_epa": 4.1, "def_epa": -16.82, "pace": 90.0,
          "off_epa_per_play": 0.058, "def_epa_per_play": -0.241,
          "success_rate": 0.438, "adj_off_epa": -0.012, "adj_def_epa": -0.384,
          "rest_days": 7, "bye": false,
          "qb_rating": 106.85, "qb_ypa": 8.19, "rb_ypc": 5.04,
          "wr_ypr": 9.58, "top3_rec_yards": 1729,
          "weekly_qb_ypa": 8.44, "weekly_qb_epa_att": 0.174,
          "weekly_rb_ypc": 3.97, "weekly_wr_ypr": 14.75,
          "weekly_top3_rec_yards": 580
        },
        "players": {
          "prior_season": 2025,
          "qb": "Josh Allen", "rb": "James Cook", "wr": "Khalil Shakir",
          "weekly": {
            "season": 2026, "week": 11,
            "qb": "J.Allen", "rb": "J.Cook", "wr": "K.Shakir"
          }
        }
      },
      "away": {
        "abbr": "MIA",
        "name": "Miami Dolphins",
        "record": "1-8",
        "win_prob": 0.1879,
        "features": {
          "off_epa": -11.4, "def_epa": -5.42, "pace": 79.7,
          "off_epa_per_play": -0.144, "def_epa_per_play": -0.071,
          "success_rate": 0.376, "adj_off_epa": -0.384, "adj_def_epa": -0.012,
          "rest_days": 7, "bye": false,
          "qb_rating": 94.91, "qb_ypa": 7.25, "rb_ypc": 5.64,
          "wr_ypr": 14.25, "top3_rec_yards": 1588,
          "weekly_qb_ypa": null, "weekly_qb_epa_att": null,
          "weekly_rb_ypc": 4.04, "weekly_wr_ypr": 14.43,
          "weekly_top3_rec_yards": 513
        },
        "players": {
          "prior_season": 2025,
          "qb": "Tua Tagovailoa", "rb": "De'Von Achane", "wr": "Jaylen Waddle",
          "weekly": {
            "season": 2026, "week": 11,
            "qb": null, "rb": "O.Gordon", "wr": "C.Douglas"
          }
        }
      },
      "predicted_winner": "BUF",
      "confidence": "high",
      "edge_pp": 31.2
    }
  ]
}
```

The `features` objects are the model's actual inputs (rolling three-game form,
EPA/play, success rate, opponent-adjusted EPA, rest, bye proxy, prior-season
player stats, and weekly player form). The `players` object gives the real
prior-season leader names, while `players.weekly` gives the current-season
leaders whose prior games feed the weekly rates. Both are shown on the Game
Insights screen. `edge_pp` is the predicted winner's distance from a 50/50 coin
flip in percentage points; it is not a market edge.

The Kaggle dataset ends at 2025, so there is no closing line for 2026 — these
are pure, temperature-calibrated model probabilities, not market-adjusted picks.
The UI therefore shows model confidence instead of fabricated spreads or
simulated scores. Forecasts are most reliable for the next upcoming week:
further-out weeks depend on games that haven't been played and therefore have no
pre-game form yet.

## Status

Data loading, EPA/play + opponent-adjusted EPA, pace/rest/bye features,
prior-season player features, current-season weekly player form, nested
walk-forward tuning, temperature calibration, generated 2026 weekly predictions,
feature-impact exports, the four-screen static UI, and offline leakage-guard
tests are complete. Run `python -m pytest` after changing the feature pipeline,
then `python -m src.model` after changing the feature set and `python -m src.predict`
whenever a new week's games are complete; refresh the served `web/` pages to see
the updated slate.
