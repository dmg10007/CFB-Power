from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    completed = predictions.dropna(subset=["home_score", "away_score"]).copy()
    if completed.empty:
        return pd.DataFrame(columns=["metric", "value"])
    home_error = completed["home_expected_points"] - completed["home_score"]
    away_error = completed["away_expected_points"] - completed["away_score"]
    projected_margin = completed["projected_spread_home"]
    actual_margin = completed["home_score"] - completed["away_score"]
    projected_total = completed["projected_total"]
    actual_total = completed["home_score"] + completed["away_score"]
    actual_home_win = (actual_margin > 0).astype(int)
    probability = completed["home_win_probability"].clip(0.001, 0.999)
    metrics = {
        "game_score_mae": float((home_error.abs().mean() + away_error.abs().mean()) / 2),
        "game_score_rmse": float(np.sqrt((np.square(home_error).mean() + np.square(away_error).mean()) / 2)),
        "spread_mae": float((projected_margin - actual_margin).abs().mean()),
        "total_mae": float((projected_total - actual_total).abs().mean()),
        "brier_score": float(np.mean(np.square(probability - actual_home_win))),
        "sample_size": float(len(completed)),
    }
    return pd.DataFrame(metrics.items(), columns=["metric", "value"])


def calibration_bins(predictions: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    completed = predictions.dropna(subset=["home_score", "away_score"]).copy()
    if completed.empty:
        return pd.DataFrame(columns=["bin", "mean_prediction", "actual_home_win_rate", "games"])
    completed["actual_home_win"] = (completed["home_score"] > completed["away_score"]).astype(int)
    completed["bin"] = pd.cut(completed["home_win_probability"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    grouped = completed.groupby("bin", observed=False).agg(
        mean_prediction=("home_win_probability", "mean"),
        actual_home_win_rate=("actual_home_win", "mean"),
        games=("game_id", "count"),
    )
    return grouped.reset_index().astype({"bin": "string"})
