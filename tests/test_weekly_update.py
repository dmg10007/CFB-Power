"""Tests for the Sunday weekly model update, calibration, and forecast pipeline."""

import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from cfb_power.cli import app
from cfb_power.weekly_update import (
    SundayUpdateResult,
    _calibrate_model,
    _evaluate_local_week,
    run_sunday_update,
)

runner = CliRunner()


def test_calibrate_model_from_games():
    games = pd.DataFrame(
        {
            "game_id": [1, 2, 3, 4],
            "completed": [True, True, True, True],
            "neutral_site": [False, False, False, False],
            "home_score": [35, 28, 24, 31],
            "away_score": [21, 24, 28, 17],
        }
    )
    eval_summary = {"margin_mae": 8.0}
    calibration = _calibrate_model(games, eval_summary)

    assert "calibrated_hfa" in calibration
    assert 0.5 <= calibration["calibrated_hfa"] <= 5.0
    assert "margin_std" in calibration
    assert calibration["sample_size"] == 4


def test_evaluate_local_week(tmp_path):
    games = pd.DataFrame(
        {
            "game_id": [101, 102],
            "week": [7, 7],
            "completed": [True, True],
            "home_score": [30, 21],
            "away_score": [20, 24],
        }
    )
    runs_dir = tmp_path / "runs" / "ds1" / "predictions_run1"
    runs_dir.mkdir(parents=True)
    preds = pd.DataFrame(
        {
            "game_id": [101, 102],
            "projected_spread_home": [7.0, 3.0],
            "projected_total": [50.0, 48.0],
            "home_win_probability": [0.70, 0.58],
        }
    )
    preds.to_csv(runs_dir / "predictions.csv", index=False)

    summary = _evaluate_local_week(tmp_path, games, week=7)
    assert summary["games_in_scope"] == 2
    assert summary["scored"] == 2
    assert summary["winner_accuracy"] == 0.5  # Game 101 correct, 102 wrong
    assert summary["margin_mae"] == pytest.approx(4.5)  # |7-10|=3, |3 - (-3)|=6 -> avg 4.5


def test_run_sunday_update_offline(tmp_path):
    import hashlib

    from cfb_power.dataset_workflow import DATASET_SCHEMA

    # Setup synthetic dataset with completed week 7 and upcoming week 8
    dataset_dir = tmp_path / "data" / "datasets" / "test_ds"
    dataset_dir.mkdir(parents=True)

    games = []
    stats = []
    # Week 7 completed games
    for gid in range(1, 5):
        games.append(
            {
                "game_id": gid,
                "season": 2026,
                "week": 7,
                "game_date": "2026-10-17T18:00:00Z",
                "home_team_id": 100 + gid,
                "away_team_id": 200 + gid,
                "home_score": 35,
                "away_score": 21,
                "neutral_site": False,
                "completed": True,
                "season_type": "regular",
            }
        )
        for tid, opp, is_h, pts, opp_pts in (
            (100 + gid, 200 + gid, True, 35, 21),
            (200 + gid, 100 + gid, False, 21, 35),
        ):
            stats.append(
                {
                    "game_id": gid,
                    "team_id": tid,
                    "opponent_id": opp,
                    "is_home": is_h,
                    "points": pts,
                    "points_allowed": opp_pts,
                    "total_yards": 400,
                    "offensive_plays": 70,
                    "passing_yards": 250,
                    "rushing_yards": 150,
                    "turnovers": 1,
                    "takeaways": 1,
                    "third_down_attempts": 12,
                    "third_down_conversions": 6,
                    "red_zone_attempts": 4,
                    "red_zone_touchdowns": 3,
                    "penalties": 4,
                    "penalty_yards": 35,
                    "time_of_possession_seconds": 1800,
                }
            )

    # Week 8 upcoming game
    games.append(
        {
            "game_id": 10,
            "season": 2026,
            "week": 8,
            "game_date": "2026-10-24T18:00:00Z",
            "home_team_id": 101,
            "away_team_id": 102,
            "home_score": np.nan,
            "away_score": np.nan,
            "neutral_site": False,
            "completed": False,
            "season_type": "regular",
        }
    )

    games_df = pd.DataFrame(games)
    stats_df = pd.DataFrame(stats)
    games_df.to_csv(dataset_dir / "games.csv", index=False)
    stats_df.to_csv(dataset_dir / "team_game_stats.csv", index=False)

    outputs = [
        {
            "file": "games.csv",
            "rows": len(games_df),
            "sha256": hashlib.sha256((dataset_dir / "games.csv").read_bytes()).hexdigest(),
        },
        {
            "file": "team_game_stats.csv",
            "rows": len(stats_df),
            "sha256": hashlib.sha256(
                (dataset_dir / "team_game_stats.csv").read_bytes()
            ).hexdigest(),
        },
    ]

    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "test_ds",
                "schema_version": DATASET_SCHEMA,
                "status": "ready",
                "outputs": outputs,
                "inputs": [],
            }
        ),
        encoding="utf-8",
    )

    result = run_sunday_update(
        data_dir=tmp_path / "data",
        season=2026,
        completed_week=7,
        upcoming_week=8,
        api_key=None,
    )

    assert isinstance(result, SundayUpdateResult)
    assert result.completed_week == 7
    assert result.upcoming_week == 8
    assert result.forecast_dir.is_dir()
    assert (result.forecast_dir / "predictions.csv").is_file()
    assert result.workbook_path.is_file()

    preds = pd.read_csv(result.forecast_dir / "predictions.csv")
    assert len(preds) == 1
    assert preds["game_id"].iloc[0] == 10
    assert preds["model"].iloc[0] == "power_rankings_v1"


def test_cli_sunday_update_command(tmp_path, monkeypatch):
    import hashlib

    from cfb_power.dataset_workflow import DATASET_SCHEMA

    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    monkeypatch.setenv("CFB_DATA_DIR", str(tmp_path / "data"))

    dataset_dir = tmp_path / "data" / "datasets" / "cli_ds"
    dataset_dir.mkdir(parents=True)

    games = [
        {
            "game_id": 1,
            "season": 2026,
            "week": 7,
            "game_date": "2026-10-17T18:00:00Z",
            "home_team_id": 10,
            "away_team_id": 20,
            "home_score": 31,
            "away_score": 24,
            "neutral_site": False,
            "completed": True,
            "season_type": "regular",
        },
        {
            "game_id": 2,
            "season": 2026,
            "week": 8,
            "game_date": "2026-10-24T18:00:00Z",
            "home_team_id": 10,
            "away_team_id": 20,
            "home_score": np.nan,
            "away_score": np.nan,
            "neutral_site": False,
            "completed": False,
            "season_type": "regular",
        },
    ]
    stats = [
        {
            "game_id": 1,
            "team_id": 10,
            "opponent_id": 20,
            "is_home": True,
            "points": 31,
            "points_allowed": 24,
            "total_yards": 410,
            "offensive_plays": 65,
            "passing_yards": 260,
            "rushing_yards": 150,
            "turnovers": 0,
            "takeaways": 2,
            "third_down_attempts": 10,
            "third_down_conversions": 5,
            "red_zone_attempts": 3,
            "red_zone_touchdowns": 2,
            "penalties": 3,
            "penalty_yards": 25,
            "time_of_possession_seconds": 1900,
        },
        {
            "game_id": 1,
            "team_id": 20,
            "opponent_id": 10,
            "is_home": False,
            "points": 24,
            "points_allowed": 31,
            "total_yards": 380,
            "offensive_plays": 62,
            "passing_yards": 240,
            "rushing_yards": 140,
            "turnovers": 2,
            "takeaways": 0,
            "third_down_attempts": 11,
            "third_down_conversions": 4,
            "red_zone_attempts": 4,
            "red_zone_touchdowns": 2,
            "penalties": 5,
            "penalty_yards": 45,
            "time_of_possession_seconds": 1700,
        },
    ]

    games_df = pd.DataFrame(games)
    stats_df = pd.DataFrame(stats)
    games_df.to_csv(dataset_dir / "games.csv", index=False)
    stats_df.to_csv(dataset_dir / "team_game_stats.csv", index=False)

    outputs = [
        {
            "file": "games.csv",
            "rows": len(games_df),
            "sha256": hashlib.sha256((dataset_dir / "games.csv").read_bytes()).hexdigest(),
        },
        {
            "file": "team_game_stats.csv",
            "rows": len(stats_df),
            "sha256": hashlib.sha256(
                (dataset_dir / "team_game_stats.csv").read_bytes()
            ).hexdigest(),
        },
    ]

    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "cli_ds",
                "schema_version": DATASET_SCHEMA,
                "status": "ready",
                "outputs": outputs,
                "inputs": [],
            }
        ),
        encoding="utf-8",
    )

    res = runner.invoke(
        app,
        [
            "sunday-update",
            "--season",
            "2026",
            "--completed-week",
            "7",
            "--upcoming-week",
            "8",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Sunday update cycle for Season 2026" in res.output
    assert "Adaptive Model Calibration" in res.output
    assert "Updated Power Rankings" in res.output
