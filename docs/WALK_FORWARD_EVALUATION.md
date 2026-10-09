# Walk-forward evaluation

## Command

```bash
python -m cfb_power.walk_forward --dataset data/datasets/DATASET_ID
```

Options: `--data-dir` (defaults to the project data directory) and `--min-train-periods` (default 2). The command reads one assembled, hash-verified dataset and writes only to a new directory, `data/runs/{dataset_id}/walk_forward_{run_id}/`. It never modifies the dataset, snapshots, or active CSVs.

## Method

Completed games are grouped into periods (season, season type, and week) and ordered by their earliest kickoff. For each period after the first `--min-train-periods`, the Ridge models are fit only on completed games from earlier periods, then predict that period's games. A period is skipped if fewer than 20 training games are available.

Feature timing is handled by the existing feature builder: it computes every rolling aggregate from a team's earlier games only (shifted one game), so a game's features never include its own result or later games. The builder runs once over the dataset; the leakage tests confirm that changing the final week's scores and statistics does not change any earlier prediction.

Two baselines are scored on the same games, using the same training window:

- `train_mean`: every home team is predicted at the training mean home score, every away team at the training mean away score, and home win probability is the training home win rate.
- `rolling_average`: each team's expected points are the average of its own season points scored and the opponent's season points allowed, falling back to the training mean when history is missing. Win probability uses a normal approximation with the standard deviation of training margins.

## Outputs

- `predictions.csv`: one row per out-of-sample game with actual scores, model and baseline predictions, and `train_games`.
- `weekly_metrics.csv`: each period and model, with game counts.
- `overall_metrics.csv`: pooled metrics per model: score MAE and RMSE, spread MAE, total MAE, Brier score, and winner accuracy.
- `calibration.csv`: pooled Ridge home-win-probability calibration bins.
- `manifest.json`: scope `out_of_sample_walk_forward`, settings, counts, tested and skipped periods, per-baseline improvement percentages, warnings, and output hashes.

In the improvement figures, a positive percentage means the Ridge model had lower error than the baseline. A negative value means the baseline was better.

## Interpretation

These are the first out-of-sample numbers in the project, but they are still limited. Early periods have thin history. A small tested sample is flagged with a warning, and period-level metrics are noisy. Results describe the data captured so far and say nothing about future seasons. A team that plays twice within one period could use its first game as pregame information for the second.

If Ridge does not beat both baselines on spread and total error, the model is not yet adding value, and that finding should drive the next modeling work.

## Not included

Upcoming-game prediction, model tuning, confidence intervals on the metrics, team display names, and dashboard views.
