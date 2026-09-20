# CFB Power V1 Plan

## Product outcome

Produce a weekly, team-stat-driven forecast package for FBS football matchups:

- Expected away and home scores.
- Model spread from the home-team perspective.
- Projected total.
- Home win probability.
- Score bands based on historical model residuals.
- A confidence label that describes forecast separation, not certainty.

The user-facing deliverable is an Excel-ready `.xlsx` export plus the flat CSV tables needed for a polished dashboard workbook.

## Architecture decisions

| Decision | V1 choice | Why |
|---|---|---|
| Scope | FBS only | Consistent competition and manageable identity normalization |
| Modeling unit | One row per game | Directly maps to forecasts and evaluation |
| Score method | Independent home/away Ridge models | Transparent, stable, and an objective baseline |
| Feature timing | Pregame rolling team metrics | Prevents data leakage |
| Storage | CSV/Parquet locally | Simple, inspectable, portable |
| Model persistence | Joblib artifact | Reproducible prediction runs |
| Excel role | Import/dashboard, not model training | Avoids fragile embedded logic and secrets |
| Validation | Season/week walk-forward in Phase 2 | Mirrors live forecasting |

## Data ingestion milestone

Create one adapter at a time. The adapter must:

1. Fetch source data using an environment-stored key where required.
2. Archive every raw payload under `data/raw/{source}/{season}/`.
3. Normalize source fields into `games.csv` and `team_game_stats.csv`.
4. Record source, retrieval UTC timestamp, and source ID in an ingestion audit log.
5. Fail closed on missing IDs, duplicate games, impossible scores, or missing opponent records.

NCAA-facing sources can be used for current team-stat validation. For longitudinal game-level modeling, retain a source that can reliably provide historical team-game statistics. Do not couple feature/model code to API response shapes.

## Feature set

Initial rolling windows: last 3, last 5, and season-to-date.

- Points scored and allowed.
- Total yards per play.
- Passing and rushing yards per offensive play.
- Turnovers and takeaways.
- Third-down conversion rate.
- Red-zone touchdown rate.
- Penalty yards.
- Time of possession.
- Home-field indicator.

The V1 feature builder shifts every team aggregation by one game. The very first game of a team-season has missing rolling values and is median-imputed by the model. That is intentional and should later be replaced by preseason priors.

## Excel dashboard specification

### Sheets supplied by export

- `README`: model version, export timestamp, and workbook instructions.
- `Predictions`: one row per matchup; primary source for a weekly slate dashboard.
- `Team_Stats`: one row per team-game with pregame rolling fields.
- `Historical_Results`: completed predictions for accuracy review.
- `Model_Performance`: MAE, RMSE, spread MAE, total MAE, Brier score, sample size.

### Dashboard workbook build

Build `workbook/CFB_Power_Dashboard.xlsx` after production data is available. Use Power Query to import the four CSV exports and create Excel Tables named `tblPredictions`, `tblTeamStats`, `tblResults`, and `tblPerformance`.

Create two dashboard views:

1. `Matchup_Dashboard`
   - Matchup selector using game ID or home/away display names.
   - Prominent projected final score.
   - Home win probability, projected spread, projected total, and score band.
   - Comparison bars for selected rolling team metrics.
   - Data freshness and model version.

2. `Weekly_Dashboard`
   - Game slate with projected score, probability, spread, total, and confidence.
   - Filters/slicers for season, week, conference once team metadata is added, confidence, and neutral site.
   - Conditional formatting for probabilities and confidence.

### Style

- Deep navy `#123047` for headers.
- Slate backgrounds and neutral table styling.
- One accent color at a time; do not use team branding by default.
- Use red/amber/green only for diagnostics or relative matchup indicators.
- Show an explicit `Data refreshed` timestamp and `model_version`.

## Acceptance criteria

- `cfb-power create-demo-data`, `cfb-power train`, and `cfb-power export-workbook` run on a clean machine after setup.
- Feature tests prove a game cannot use its own final stats in rolling averages.
- Model tests verify score, spread, total, and probability outputs.
- Workbook export opens in Excel and contains the five required sheets.
- Generated files are ignored by Git.
- No credentials or raw production data are committed.

## Milestones

1. Foundation — package, schemas, validation, demo data, baseline models, exports, tests. **Included in initial scaffold.**
2. Ingestion — NCAA/CFBD adapter, raw archival, normalization, identity map, ingestion audit.
3. Walk-forward backtesting — season/week splits, persisted prediction snapshots, calibration report.
4. Dashboard — Power Query connections, polished matchup and weekly sheets, printable weekly report.
5. V1.1 — strength of schedule/Elo, preseason priors, robust score simulation, schedule automation.

## Immediate next work

- Verify scaffold commands and tests locally.
- Select and document an API source/credentials policy.
- Implement a normalized ingestion adapter using a sample season.
- Reconcile team IDs and renamed programs across historical seasons.
- Replace demo data with source-validated historical data before interpreting performance metrics.
