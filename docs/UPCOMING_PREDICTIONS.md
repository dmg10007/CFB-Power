# Upcoming-game predictions

## Workflow

1. Capture the current week while its games are still upcoming. Games with a confirmed kickoff time are kept as upcoming; games already completed are kept as history.

```bash
python -m cfb_power.ingestion.cfbd_batch --season 2026 --season-type regular --week 7
```

2. Assemble every snapshot into a new dataset and forecast from it:

```bash
python -m cfb_power.dataset_workflow assemble --all
python -m cfb_power.upcoming --dataset data/datasets/DATASET_ID
```

Options: `--data-dir`, `--as-of` (ISO timestamp with a timezone; defaults to now in UTC), and `--min-history-games` (default 3). Output goes to a new `data/runs/{dataset_id}/predictions_{run_id}/` directory containing `predictions.csv`, `skipped.csv`, `upcoming_predictions.xlsx`, and `manifest.json`. The dataset is never modified.

## Model

The forecaster is the rolling-average model: each side's expected points are the average of its own season points scored and the opponent's season points allowed, using only completed games played before the upcoming game's kickoff. A team with no history falls back to the league mean from completed games. Win probability is a normal approximation of the projected margin using the spread of completed-game margins. There is no home-field term: adding one made out-of-sample results worse in the model experiments, and neutral-site games therefore need no special handling.

This model was chosen because it had the lowest spread error and Brier score among the models walk-forward-tested so far. On 171 out-of-sample games (weeks 3-5 of 2026) its average spread error was about 14.3 points and its Brier score was 0.204. Treat forecasts as rough guides: errors of two touchdowns on the spread are normal, and the evidence base is three weeks.

## Safety rules

- Only games with a confirmed kickoff after the as-of time are forecast. Games that have started, are in progress, or have scores but are not marked completed are listed in `skipped.csv` with a reason.
- Each forecast shows how many prior games each team has played and a `low_history` flag when either team has fewer than the minimum.
- A warning is recorded when the newest completed game is more than 9 days older than the as-of time, which usually means a recent week has not been captured.
- Team names are read from the hash-verified raw schedule archives of the dataset's input snapshots. If an archive is missing or changed, names are left blank with a warning and forecasts are still produced with team IDs.
- The command fails without writing a run if the dataset has no completed games or no forecastable games.

## Limits

No confidence intervals are produced. The forecast does not use injuries, weather, betting lines, or opponent adjustment, and early-season history is thin. Re-run the walk-forward evaluation as more weeks are captured and check that the rolling average still earns its place.
