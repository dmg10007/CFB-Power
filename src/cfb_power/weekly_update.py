"""Sunday weekly update pipeline: evaluate completed week against actuals, calibrate model, and forecast upcoming slate."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Paths
from .dataset_workflow import discover_snapshots
from .exports import export_all
from .ingestion.cfbd_batch import ingest_cfbd_batch
from .ingestion.cfbd_client import CfbdClient
from .power_ranking import PowerRankings, fit_power_rankings
from .results_report import generate_report
from .revisions import assemble_or_review
from .team_names import resolve_team_names
from .upcoming import predict_upcoming


@dataclass
class SundayUpdateResult:
    """Summary of the Sunday model evaluation and upcoming forecast generation."""

    season: int
    completed_week: int
    upcoming_week: int
    evaluation: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    dataset_dir: Path = field(default_factory=Path)
    forecast_dir: Path = field(default_factory=Path)
    report_dir: Path | None = None
    workbook_path: Path = field(default_factory=Path)
    rankings_count: int = 0
    top_teams: list[dict[str, Any]] = field(default_factory=list)


def run_sunday_update(
    data_dir: Path | str,
    season: int,
    completed_week: int,
    upcoming_week: int,
    api_key: str | None = None,
    emit: Callable[[str], None] | None = None,
    client: CfbdClient | None = None,
    min_history_games: int = 3,
    offline: bool = False,
) -> SundayUpdateResult:
    """Execute the full Sunday update cycle:

    1. Ingest completed week's scores & box stats and upcoming week's schedule (if API key provided and not offline)
    2. Assemble and activate the verified dataset
    3. Evaluate previous pregame forecasts against actual outcomes (Accuracy, MAE, Brier)
    4. Calibrate model parameters (empirical HFA, residual error scale) and refit Power Rankings
    5. Generate new upcoming forecasts for the next week using opponent-adjusted Power Rankings
    6. Export formatted Excel workbook and activate processed pipeline data
    """
    data_path = Path(data_dir)
    paths = Paths(data_dir=data_path)
    paths.ensure()
    logger = emit or (lambda msg: None)

    # 1. Ingestion & Dataset Assembly
    dataset_dir: Path | None = None
    if not offline and (api_key or client):
        if client is None:
            client = CfbdClient(api_key=api_key)  # type: ignore[arg-type]
        try:
            logger(f"Ingesting completed Week {completed_week} results and box scores...")
            ingest_cfbd_batch(
                season=season,
                season_type="regular",
                week=completed_week,
                data_dir=data_path,
                client=client,
            )
            logger(f"Ingesting upcoming Week {upcoming_week} schedule...")
            ingest_cfbd_batch(
                season=season,
                season_type="regular",
                week=upcoming_week,
                data_dir=data_path,
                client=client,
            )
            logger("Assembling verified dataset snapshots...")
            snapshots = discover_snapshots(data_path)
            dataset_dir = assemble_or_review(snapshots, data_path)
        except Exception as exc:
            logger(f"⚠️ Live ingestion failed: {exc}")
            datasets_dir = data_path / "datasets"
            candidates = (
                sorted(
                    [
                        d
                        for d in datasets_dir.iterdir()
                        if d.is_dir() and (d / "games.csv").is_file()
                    ],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if datasets_dir.is_dir()
                else []
            )
            if candidates:
                logger(f"Falling back to existing local dataset: {candidates[0].name}")
                dataset_dir = candidates[0]
            else:
                raise

    if dataset_dir is None:
        # Offline mode: use latest verified dataset or processed data
        datasets_dir = data_path / "datasets"
        candidates = (
            sorted(
                [d for d in datasets_dir.iterdir() if d.is_dir() and (d / "games.csv").is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if datasets_dir.is_dir()
            else []
        )
        if not candidates:
            raise ValueError("No verified datasets found for Sunday update.")
        dataset_dir = candidates[0]

    # Activate dataset into data/processed/
    games_src = dataset_dir / "games.csv"
    stats_src = dataset_dir / "team_game_stats.csv"
    games_dst = paths.processed_dir / "games.csv"
    stats_dst = paths.processed_dir / "team_game_stats.csv"
    games_dst.write_bytes(games_src.read_bytes())
    stats_dst.write_bytes(stats_src.read_bytes())

    games_df = pd.read_csv(games_dst)
    stats_df = pd.read_csv(stats_dst)
    team_names = resolve_team_names(data_path)

    # 2. Evaluate Completed Week Predictions vs Actual Results
    logger(f"Evaluating model predictions vs actual results for Week {completed_week}...")
    report_dir = None
    eval_summary: dict[str, Any] = {}

    if client is not None:
        try:
            report_dir = generate_report(
                data_dir=data_path,
                season=season,
                week=completed_week,
                season_type="regular",
                client=client,
            )
            summary_csv = report_dir / "summary.csv"
            if summary_csv.is_file():
                summary_df = pd.read_csv(summary_csv)
                all_row = summary_df[summary_df["group"] == "all"]
                if not all_row.empty:
                    eval_summary = {
                        "games_in_scope": int(all_row["games"].iloc[0]),
                        "scored": int(all_row["scored"].iloc[0]),
                        "winner_accuracy": float(all_row["winner_accuracy"].iloc[0]),
                        "margin_mae": float(all_row["margin_mae"].iloc[0]),
                        "total_mae": float(all_row["total_mae"].iloc[0]),
                        "brier_score": float(all_row["brier_score"].iloc[0]),
                    }
        except Exception as exc:
            logger(f"Live results report note: {exc}")

    # Fallback / Local evaluation if live report wasn't generated
    if not eval_summary:
        eval_summary = _evaluate_local_week(data_path, games_df, completed_week)

    # 3. Model Calibration Based on Actual Results
    logger("Calibrating model parameters and updating Power Rankings...")
    calibration = _calibrate_model(games_df, eval_summary)

    # Fit updated Power Rankings incorporating all completed games through completed_week
    completed_games = games_df[games_df["completed"].astype(bool)].copy()
    rankings: PowerRankings = fit_power_rankings(
        completed_games,
        stats_df,
        team_names=team_names,
        home_advantage=calibration["calibrated_hfa"],
    )
    rankings_df = rankings.to_dataframe()

    # 4. Generate Upcoming Slate Predictions using Power Rankings
    logger(f"Generating opponent-adjusted Power Rankings forecasts for Week {upcoming_week}...")
    forecast_dir = predict_upcoming(
        dataset_dir=dataset_dir,
        data_dir=data_path,
        min_history_games=min_history_games,
        model="power_rankings",
    )

    # 5. Export Updated Workbook Package
    logger("Exporting updated Excel workbook and summary outputs...")
    predictions_df = pd.read_csv(forecast_dir / "predictions.csv")
    historical_df = completed_games.copy()
    performance_df = pd.DataFrame([eval_summary]) if eval_summary else pd.DataFrame()

    export_all(
        predictions=predictions_df,
        team_stats=stats_df,
        historical_results=historical_df,
        performance=performance_df,
        paths=paths,
        power_rankings=rankings_df,
        team_names=team_names,
    )

    top_teams = rankings_df.head(10)[["rank", "team_name", "power_rating", "tier"]].to_dict(
        orient="records"
    )

    logger(f"Sunday update complete! New forecasts published to {forecast_dir.name}")

    return SundayUpdateResult(
        season=season,
        completed_week=completed_week,
        upcoming_week=upcoming_week,
        evaluation=eval_summary,
        calibration=calibration,
        dataset_dir=dataset_dir,
        forecast_dir=forecast_dir,
        report_dir=report_dir,
        workbook_path=paths.workbook_xlsx,
        rankings_count=len(rankings_df),
        top_teams=top_teams,
    )


def _calibrate_model(games_df: pd.DataFrame, eval_summary: dict[str, Any]) -> dict[str, Any]:
    """Calculate adaptive model calibration updates from season actuals and prediction errors."""
    completed = games_df[games_df["completed"].astype(bool)]
    if completed.empty:
        return {"calibrated_hfa": 2.5, "margin_std": 13.5, "sample_size": 0}

    # 1. Calibrate empirical Home Field Advantage (HFA)
    non_neutral = completed[~completed["neutral_site"].astype(bool)]
    if len(non_neutral) > 0:
        actual_hfa_mean = float((non_neutral["home_score"] - non_neutral["away_score"]).mean())
        # Shrink toward classical +2.5 prior with sample size weighting
        n = len(non_neutral)
        prior_weight = 10.0 / (10.0 + n)
        calibrated_hfa = round(prior_weight * 2.5 + (1.0 - prior_weight) * actual_hfa_mean, 2)
    else:
        calibrated_hfa = 2.5

    # 2. Calibrate score margin error variance (residual std)
    # If we have actual MAE from predictions, use normal conversion (std ~ MAE * sqrt(pi/2))
    if "margin_mae" in eval_summary and eval_summary["margin_mae"] > 0:
        emp_margin_std = round(eval_summary["margin_mae"] * np.sqrt(np.pi / 2.0), 2)
    else:
        margins = completed["home_score"] - completed["away_score"]
        emp_margin_std = round(float(margins.std(ddof=1)), 2) if len(completed) > 1 else 13.5

    return {
        "calibrated_hfa": max(0.5, min(calibrated_hfa, 5.0)),
        "margin_std": max(8.0, min(emp_margin_std, 20.0)),
        "sample_size": len(completed),
        "actual_home_margin_avg": round(
            float((non_neutral["home_score"] - non_neutral["away_score"]).mean()), 2
        )
        if len(non_neutral) > 0
        else 2.5,
    }


def _evaluate_local_week(data_dir: Path, games_df: pd.DataFrame, week: int) -> dict[str, Any]:
    """Find local candidate predictions for week and evaluate against completed scores."""
    week_games = games_df[(games_df["week"] == week) & games_df["completed"].astype(bool)]
    if week_games.empty:
        return {}

    # Check for past predictions in runs/
    runs_dir = data_dir / "runs"
    pred_files = sorted(runs_dir.glob("*/predictions_*/predictions.csv"), reverse=True)
    if not pred_files:
        return {"games_in_scope": len(week_games), "scored": 0, "winner_accuracy": 0.0}

    # Look for matching game IDs
    week_gids = set(week_games["game_id"])
    matches = []
    for p_file in pred_files:
        try:
            preds = pd.read_csv(p_file)
            overlap = preds[preds["game_id"].isin(week_gids)]
            if not overlap.empty:
                matches.append(overlap)
                if len(overlap) >= len(week_games) * 0.7:
                    break
        except Exception:
            continue

    if not matches:
        return {"games_in_scope": len(week_games), "scored": 0, "winner_accuracy": 0.0}

    combined_preds = pd.concat(matches).drop_duplicates(subset=["game_id"], keep="first")
    merged = week_games.merge(
        combined_preds[
            ["game_id", "projected_spread_home", "projected_total", "home_win_probability"]
        ],
        on="game_id",
        how="inner",
    )
    if merged.empty:
        return {"games_in_scope": len(week_games), "scored": 0, "winner_accuracy": 0.0}

    actual_margin = merged["home_score"] - merged["away_score"]
    actual_total = merged["home_score"] + merged["away_score"]
    predicted_home_win = merged["projected_spread_home"] > 0
    actual_home_win = actual_margin > 0

    winner_correct = (predicted_home_win == actual_home_win) & (actual_margin != 0)
    margin_errors = (merged["projected_spread_home"] - actual_margin).abs()
    total_errors = (merged["projected_total"] - actual_total).abs()
    brier = (merged["home_win_probability"] - actual_home_win.astype(int)) ** 2

    return {
        "games_in_scope": len(week_games),
        "scored": len(merged),
        "winner_accuracy": round(float(winner_correct.mean()), 3),
        "margin_mae": round(float(margin_errors.mean()), 2),
        "total_mae": round(float(total_errors.mean()), 2),
        "brier_score": round(float(brier.mean()), 3),
        "margin_bias": round(float((actual_margin - merged["projected_spread_home"]).mean()), 2),
    }
