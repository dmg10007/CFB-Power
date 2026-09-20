# CFB Power

Team-stat-driven college-football score forecasts with a reproducible Python pipeline and Excel-ready outputs.

## V1 objective

For each FBS matchup, predict expected home and away points using only information available before kickoff. Export a digestible workbook data package containing:

- `predictions.csv` — projected scores, spread, total, home win probability, score intervals, and confidence tier.
- `team_stats.csv` — pregame rolling team metrics used in the matchup.
- `historical_results.csv` — locked historical predictions compared with final scores.
- `model_performance.csv` — score, spread, total, and probability-quality metrics.
- `cfb_forecast_export.xlsx` — a formatted multi-sheet Excel workbook suitable for dashboard construction and refresh.

V1 deliberately prioritizes team-level statistics and transparent validation. It does not use betting lines, injuries, weather, player projections, or in-game data.

## Architecture

```text
canonical games + team-game statistics
             ↓
validation and pregame rolling features
             ↓
home-score model + away-score model
             ↓
projected score / spread / total / win probability / intervals
             ↓
CSV + Excel exports
```

The package does not scrape or call a sports API at import time. API clients belong in an ingestion adapter and raw responses should be saved before normalization. This keeps the model reproducible and prevents credentials from entering the workbook.

## Data contract

Place normalized files in `data/processed/`:

- `games.csv`: `game_id, season, week, game_date, home_team_id, away_team_id, home_score, away_score, neutral_site, completed`
- `team_game_stats.csv`: `game_id, team_id, opponent_id, is_home, points, points_allowed, total_yards, offensive_plays, passing_yards, rushing_yards, turnovers, takeaways, third_down_attempts, third_down_conversions, red_zone_attempts, red_zone_touchdowns, penalties, penalty_yards, time_of_possession_seconds`

Use stable, source-owned team IDs—not display names—as keys. The feature layer joins game context to team stats and shifts rolling averages so a game never sees its own outcome.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate         # Linux/macOS
# .venv\\Scripts\\Activate.ps1     # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Copy `.env.example` to `.env` only when an ingestion adapter requires a key. Do not commit `.env`.

## First runnable demo

```bash
cfb-power create-demo-data
cfb-power train
cfb-power export-workbook
pytest
```

The demo creates synthetic historical data for pipeline verification only; it is not football evidence. Replace it with validated normalized NCAA/CFBD-derived data before evaluating model quality.

## Weekly workflow

1. Ingest source data and archive raw responses.
2. Normalize and validate completed games plus team-game stats.
3. Run `cfb-power train` to backtest and create predictions.
4. Run `cfb-power export-workbook`.
5. In Excel, use Data → Refresh All if the dashboard workbook is connected to the CSV exports.
6. Preserve dated outputs; never overwrite historical prediction snapshots.

## Modeling method

- Two regularized Ridge regression models forecast home and away points independently.
- Pregame rolling metrics use prior games only and include 3-game, 5-game, and season-to-date windows.
- Matchup features pair home offense with away defense, and away offense with home defense.
- Residual standard deviation defines initial score intervals.
- A normal approximation to projected margin produces an initial home win probability.
- Evaluation reports MAE, RMSE, spread MAE, total MAE, Brier score, and calibration bins.

This is the transparent baseline. Gradient boosting, Elo/strength-of-schedule, and simulation belong in later versions after the data contract and walk-forward process are proven.

## Project layout

```text
src/cfb_power/        Pipeline package
data/raw/             Archived API responses; ignored by Git
data/processed/       Normalized source tables; ignored by Git
data/exports/         Generated workbook/CSVs; ignored by Git
models/                Serialized model artifacts; ignored by Git
tests/                 Unit tests for no-leakage and output contracts
docs/                  Planning and operating documentation
```

## Quality rules

- All model features must be available before kickoff.
- Never use random row splitting for time-series evaluation.
- Keep a schema/version record with each exported prediction set.
- Treat public API wrappers as adapters, not the system of record.
- Do not embed secret API keys in source code, Excel, or committed configuration.

See `docs/V1_PLAN.md` for implementation milestones and Excel dashboard specification.
