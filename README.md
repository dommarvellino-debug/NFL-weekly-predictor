# NFL Weekly Game Predictor

Predict the outcomes of weekly NFL games using historical play-by-play and
schedule data from the [nflverse](https://nflverse.nflverse.com/) via
`nflreadpy`, enriched with historical scores and betting lines from the Kaggle
[NFL Scores & Betting Data](https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data)
dataset, plus prior-season team player leaders from the Kaggle
[NFL Stats 1999-2022](https://www.kaggle.com/datasets/philiphyde1/nfl-stats-1999-2022)
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
│   ├── player_stats.py  # Kaggle player-stat loader + team leaders
│   ├── features.py   # EPA / pace / rest / bye / player form + rolling form
│   ├── model.py      # logistic regression training + walk-forward evaluation
│   └── predict.py    # current-week predictions + web data exports
├── tests/            # unit tests
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
- **Model Metrics** — walk-forward accuracy, Brier score, benchmarks, player-stat
  coverage/feature impact, and a reliability diagram built from out-of-fold
  predictions.
- **Game Insights** — switch between games to inspect the exact rolling EPA,
  pace, rest, bye, and prior-season player inputs behind each probability.

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

## Feature availability

| Feature | Source | Implementation |
| --- | --- | --- |
| Offensive EPA | nflreadpy PBP | `epa` summed/averaged where `posteam` = offense |
| Defensive EPA | nflreadpy PBP | `epa` where `defteam` = defense (EPA allowed; higher = worse) |
| Pace | nflreadpy PBP | offensive plays per game |
| Rest days | nflreadpy schedules | `home_rest` / `away_rest` (days since previous game) |
| Bye week | derived | `rest >= 12` days as a proxy |
| Prior-season player form | Kaggle player stats | leading QB rating/YPA, lead RB YPC, lead receiver YPR, top-3 receiving yards |
| Spread / over-under / weather | Kaggle | joined on season + date + teams (2024: 282/285 games) |

The scores/betting Kaggle data does **not** contain EPA or pace; nflreadpy covers
those. The player-stat Kaggle dataset adds entering-season player form. For a
game in season `S`, only player leaders from season `S-1` are joined, so the
model never sees a team's current-season player production before predicting
that season's games. Team IDs `LAR`/`LVR` in Kaggle are mapped to nflverse's
`LA`/`LV`.

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
- prior-season QB rating/YPA, lead RB YPC, lead receiver YPR and top-3 receiving
  yards for each side, plus home-minus-away differences (15 player inputs)
- one-hot team identity (lets the model learn team strength)

That is **98 features** in total (30 numeric + 68 team indicators). Same-game
EPA/pace is excluded to avoid leaking the result, and player stats are always
from the prior season. The Kaggle closing spread is held out of the model and
used only as a market baseline.

```powershell
python -m src.model
```

This prints time-ordered validation and saves `models/logistic_regression.joblib`
plus `models/logistic_regression.json` (metrics). Validation is walk-forward —
each season is scored by a model trained only on earlier seasons:

| Test season | Accuracy | ROC AUC | Brier |
| --- | --- | --- | --- |
| 2021 | 0.630 | 0.628 | 0.248 |
| 2022 | 0.550 | 0.596 | 0.250 |
| 2023 | 0.593 | 0.611 | 0.244 |
| 2024 | 0.632 | 0.687 | 0.222 |
| 2025 | 0.620 | 0.673 | 0.228 |

Last-three-season average: **0.615 accuracy / 0.657 AUC / 0.231 Brier**, versus
baselines of 0.550 for always picking the home team and 0.667 for the closing
market favorite. The final model refit on all ten seasons scores 0.652 accuracy
and 0.706 AUC in-sample; that number is descriptive, not a validation estimate.
The model beats the naive baseline but not the betting market, which is a useful
reference point rather than a surprise.

Training also writes a calibration table from all 1,420 out-of-fold walk-forward
predictions, using fixed 0–10% through 90–100% probability bins. Those values are
stored in both `models/logistic_regression.json` and
`web/data/model-metrics.json` and power the reliability diagram on the Model
Metrics screen.

Training also exports standardized logistic coefficients (`feature_importance`)
and the player-only subset (`player_feature_importance`). The largest player
inputs in the final fit are QB yards/attempt difference (+0.2501) and top-3
receiving-yard difference (+0.2232). The Model Metrics screen displays the
player-stat coverage and these coefficients; Game Insights displays the actual
leaders and their numbers for both teams.

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
    "features": 98,
    "player_features": 15,
    "player_source": "philiphyde1/nfl-stats-1999-2022",
    "player_seasons": [2012, 2025],
    "walk_forward_accuracy": 0.6148,
    "walk_forward_auc": 0.6568
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
        "win_prob": 0.8996,
        "features": {
          "off_epa": 4.1, "def_epa": -16.82, "pace": 90.0,
          "rest_days": 7, "bye": false,
          "qb_rating": 106.85, "qb_ypa": 8.19, "rb_ypc": 5.04,
          "wr_ypr": 9.58, "top3_rec_yards": 1729
        },
        "players": {
          "prior_season": 2025,
          "qb": "Josh Allen", "rb": "James Cook", "wr": "Khalil Shakir"
        }
      },
      "away": {
        "abbr": "MIA",
        "name": "Miami Dolphins",
        "record": "1-8",
        "win_prob": 0.1004,
        "features": {
          "off_epa": -11.4, "def_epa": -5.42, "pace": 79.7,
          "rest_days": 7, "bye": false,
          "qb_rating": 94.91, "qb_ypa": 7.25, "rb_ypc": 5.64,
          "wr_ypr": 14.25, "top3_rec_yards": 1588
        },
        "players": {
          "prior_season": 2025,
          "qb": "Tua Tagovailoa", "rb": "De'Von Achane", "wr": "Jaylen Waddle"
        }
      },
      "predicted_winner": "BUF",
      "confidence": "high",
      "edge_pp": 40.0
    }
  ]
}
```

The `features` objects are the model's actual inputs (rolling three-game form,
rest, bye proxy, and prior-season player stats), and `players` gives the real
leader names behind those numbers. Both are shown on the Game Insights screen.
`edge_pp` is the predicted winner's distance from a 50/50 coin flip in
percentage points; it is not a market edge.

The Kaggle dataset ends at 2025, so there is no closing line for 2026 — these
are pure model probabilities, not market-adjusted picks. The UI therefore shows
model confidence instead of fabricated spreads or simulated scores. Forecasts
are most reliable for the next upcoming week: further-out weeks depend on games
that haven't been played and therefore have no pre-game form yet.

## Status

Data loading, EPA/pace/rest/bye features, prior-season player features, the
logistic-regression model, generated 2026 weekly predictions, calibration and
feature-impact exports, and the four-screen static UI are complete. Run
`python -m src.model` after changing the feature set, then `python -m src.predict`
whenever a new week's games are complete; refresh the served `web/` pages to see
the updated slate.
