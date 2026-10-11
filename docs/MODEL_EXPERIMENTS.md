# Model experiments

## Purpose

The first walk-forward run showed the baseline Ridge model losing to a simple rolling-average baseline. This change adds a small set of candidate models so that fixes can be compared on the same out-of-sample games. Candidates are experimental: they are never used for production forecasts or exports.

## Command

```bash
python -m cfb_power.walk_forward --dataset data/datasets/DATASET_ID --with-candidates
```

Without `--with-candidates` the behavior and output are unchanged. With it, five extra models are scored beside `ridge`, `train_mean` and `rolling_average`, on exactly the same games and training windows.

## Candidates

| Model | Idea |
|---|---|
| `ridge_scaled_all` | Ridge (alpha 10) on standardized versions of every existing feature. Isolates the effect of scaling. |
| `ridge_scaled_small` | Ridge (alpha 10) on standardized season points, points allowed, yards per play, and home field. Tests fewer features plus scaling. |
| `shrunk_k2` | Each team's season points and points-allowed averages are pulled toward the league mean, as if it had 2 extra games at the league average. |
| `shrunk_k5` | The same with a prior weight of 5 games. |
| `shrunk_k5_home` | `shrunk_k5` plus a home-field adjustment estimated from training games and applied only to non-neutral games. |

Shrinkage works per team: the estimate is (games played x team average + k x league mean) / (games played + k). A team with no history sits at the league mean, and with k = 0 the method reproduces the rolling-average baseline exactly (a unit test checks this). The `shrunk_k5_home` candidate differs from `shrunk_k5` only by the home-field term, so the two show what home field adds.

## Reading the results

`overall_metrics.csv` gains one row per candidate. `manifest.json` adds `spread_mae`, `total_mae`, and `brier_score` improvements for each candidate against `rolling_average`; positive means the candidate had lower error. The manifest also lists candidate definitions.

## Cautions

- Five candidates are scored at once on one small sample. The best-looking one may simply be the luckiest. The prior weights and alpha were fixed in advance and are not tuned on these games, but picking a winner after looking is still a form of selection.
- Treat a candidate as promising only if it beats the rolling average on spread error and Brier score, and confirm it on later weeks as they are captured.
- A candidate beating Ridge but not the rolling average is not an improvement over the simplest option.
- Results describe the data captured so far and say nothing about future seasons.

## Not included

Hyperparameter search, opponent-adjusted ratings, using a candidate for exports or upcoming-game predictions, and confidence intervals.
