"""Walk-forward evaluation tests using synthetic verified datasets; no API key needed."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cfb_power.dataset_workflow import DATASET_SCHEMA, DatasetError
from cfb_power.ingestion.cfbd_canonical import PLAY_COUNT_BASIS
from cfb_power.schemas import GAME_COLUMNS, TEAM_GAME_STAT_COLUMNS
from cfb_power.walk_forward import MODEL_NAMES, main, walk_forward_evaluate

TEAMS = list(range(101, 117))


def make_tables(weeks=6, seed=11):
    rng = np.random.default_rng(seed)
    games, stats, game_id = [], [], 1
    for week in range(1, weeks + 1):
        order = [int(team) for team in rng.permutation(TEAMS)]
        kickoff = pd.Timestamp("2026-09-05T16:00:00Z") + pd.Timedelta(days=7 * (week - 1))
        for index in range(0, len(order), 2):
            home, away = order[index], order[index + 1]
            points = {home: int(rng.integers(10, 45)), away: int(rng.integers(10, 45))}
            games.append({
                "game_id": game_id, "season": 2026, "week": week, "game_date": kickoff.isoformat(),
                "home_team_id": home, "away_team_id": away,
                "home_score": points[home], "away_score": points[away],
                "neutral_site": False, "completed": True, "season_type": "regular",
            })
            for team, opponent, is_home in ((home, away, True), (away, home, False)):
                plays = int(rng.integers(55, 80))
                passing, rushing = int(rng.integers(120, 300)), int(rng.integers(50, 200))
                stats.append({
                    "game_id": game_id, "team_id": team, "opponent_id": opponent, "is_home": is_home,
                    "points": points[team], "points_allowed": points[opponent],
                    "total_yards": passing + rushing, "offensive_plays": plays,
                    "passing_yards": passing, "rushing_yards": rushing,
                    "turnovers": int(rng.integers(0, 4)), "takeaways": int(rng.integers(0, 4)),
                    "third_down_attempts": 12, "third_down_conversions": int(rng.integers(2, 10)),
                    "red_zone_attempts": np.nan, "red_zone_touchdowns": np.nan,
                    "penalties": int(rng.integers(2, 10)), "penalty_yards": int(rng.integers(15, 90)),
                    "time_of_possession_seconds": int(rng.integers(1500, 2100)),
                    "offensive_plays_basis": PLAY_COUNT_BASIS,
                })
            game_id += 1
    return pd.DataFrame(games, columns=[*GAME_COLUMNS, "season_type"]), pd.DataFrame(
        stats, columns=[*TEAM_GAME_STAT_COLUMNS, "offensive_plays_basis"]
    )


def write_dataset(tmp_path, games, stats, name="ds1"):
    directory = tmp_path / "datasets" / name
    directory.mkdir(parents=True)
    outputs = []
    for filename, frame in (("games.csv", games), ("team_game_stats.csv", stats)):
        frame.to_csv(directory / filename, index=False)
        outputs.append({
            "file": filename, "rows": len(frame),
            "sha256": hashlib.sha256((directory / filename).read_bytes()).hexdigest(),
        })
    (directory / "manifest.json").write_text(json.dumps({
        "dataset_id": name, "schema_version": DATASET_SCHEMA, "status": "ready", "outputs": outputs,
    }))
    return directory


def shift_last_week(games, stats, week):
    games, stats = games.copy(), stats.copy()
    mask = games["week"] == week
    games.loc[mask, ["home_score", "away_score"]] += 25
    scores = games[mask].set_index("game_id")
    for index, row in stats[stats["game_id"].isin(scores.index)].iterrows():
        game = scores.loc[row["game_id"]]
        own, other = ("home_score", "away_score") if row["is_home"] else ("away_score", "home_score")
        stats.loc[index, ["points", "points_allowed"]] = [game[own], game[other]]
        stats.loc[index, "total_yards"] += 400
        stats.loc[index, "passing_yards"] += 400
    return games, stats


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_walk_forward_scores_only_later_periods_out_of_sample(tmp_path):
    games, stats = make_tables()
    dataset = write_dataset(tmp_path, games, stats)
    before = {path: digest(path) for path in dataset.iterdir()}
    run = walk_forward_evaluate(dataset, tmp_path)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["evaluation_scope"] == "out_of_sample_walk_forward"
    assert manifest["counts"] == {"periods": 6, "tested_periods": 3, "skipped_periods": 3, "tested_games": 24}
    assert [item["train_games"] for item in manifest["tested"]] == [24, 32, 40]
    assert [item["period"] for item in manifest["tested"]] == [f"2026 regular week {w}" for w in (4, 5, 6)]
    assert {item["period"] for item in manifest["skipped"]} == {f"2026 regular week {w}" for w in (1, 2, 3)}
    assert any("small sample" in warning for warning in manifest["warnings"])
    predictions = pd.read_csv(run / "predictions.csv")
    assert len(predictions) == 24 and set(predictions["train_games"]) == {24, 32, 40}
    weekly = pd.read_csv(run / "weekly_metrics.csv")
    assert len(weekly) == 9 and set(weekly["model"]) == set(MODEL_NAMES)
    for artifact in manifest["outputs"]:
        assert digest(run / artifact["file"]) == artifact["sha256"]
    assert {path: digest(path) for path in dataset.iterdir()} == before
    assert run.parent == tmp_path / "runs" / "ds1"


def test_baselines_and_metrics_are_computed_from_training_data_only(tmp_path):
    games, stats = make_tables()
    run = walk_forward_evaluate(write_dataset(tmp_path, games, stats), tmp_path)
    predictions = pd.read_csv(run / "predictions.csv")
    first = predictions[predictions["period"] == "2026 regular week 4"]
    training = games[games["week"] <= 3]
    assert first["train_mean_home_points"].iloc[0] == pytest.approx(training["home_score"].mean())
    assert first["train_mean_away_points"].iloc[0] == pytest.approx(training["away_score"].mean())
    expected_rate = ((training["home_score"] > training["away_score"]).mean())
    assert first["train_mean_home_win_probability"].iloc[0] == pytest.approx(np.clip(expected_rate, 0.001, 0.999))
    overall = pd.read_csv(run / "overall_metrics.csv").set_index("model")
    for name in MODEL_NAMES:
        margin_error = (
            (predictions[f"{name}_home_points"] - predictions[f"{name}_away_points"])
            - (predictions["home_score"] - predictions["away_score"])
        ).abs().mean()
        assert overall.loc[name, "spread_mae"] == pytest.approx(margin_error)
        assert overall.loc[name, "games"] == 24
    assert predictions["ridge_home_win_probability"].between(0, 1).all()


def test_later_results_never_change_earlier_predictions(tmp_path):
    games, stats = make_tables()
    first = pd.read_csv(walk_forward_evaluate(write_dataset(tmp_path, games, stats, "a"), tmp_path) / "predictions.csv")
    changed_games, changed_stats = shift_last_week(games, stats, 6)
    second = pd.read_csv(
        walk_forward_evaluate(write_dataset(tmp_path, changed_games, changed_stats, "b"), tmp_path) / "predictions.csv"
    )
    prediction_columns = [column for column in first.columns if column.startswith(MODEL_NAMES)]
    pd.testing.assert_frame_equal(first[["game_id", *prediction_columns]], second[["game_id", *prediction_columns]])
    assert not first["home_score"].equals(second["home_score"])


def test_future_games_are_never_in_the_training_window(tmp_path):
    games, stats = make_tables()
    run = walk_forward_evaluate(write_dataset(tmp_path, games, stats), tmp_path)
    manifest = json.loads((run / "manifest.json").read_text())
    weeks = {item["period"]: int(item["period"].split()[-1]) for item in manifest["tested"]}
    for item in manifest["tested"]:
        assert int(item["train_through"].split()[-1]) == weeks[item["period"]] - 1
        assert item["train_games"] == 8 * (weeks[item["period"]] - 1)


def test_insufficient_history_fails_without_creating_a_run(tmp_path):
    games, stats = make_tables(weeks=3)
    dataset = write_dataset(tmp_path, games, stats)
    with pytest.raises(DatasetError, match="no evaluable periods"):
        walk_forward_evaluate(dataset, tmp_path)
    assert not (tmp_path / "runs").exists()


def test_invalid_settings_and_untrusted_datasets_are_rejected(tmp_path):
    games, stats = make_tables()
    dataset = write_dataset(tmp_path, games, stats)
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            walk_forward_evaluate(dataset, tmp_path, value)
    with (dataset / "games.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(DatasetError, match="hash mismatch"):
        walk_forward_evaluate(dataset, tmp_path)
    assert not (tmp_path / "runs").exists()


def test_cli_reports_out_of_sample_results_and_failures(tmp_path, capsys):
    games, stats = make_tables()
    dataset = write_dataset(tmp_path, games, stats)
    assert main(["--dataset", str(dataset), "--data-dir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "Out-of-sample games: 24 over 3 periods" in output and "WARNING: small sample" in output
    assert main(["--dataset", str(tmp_path / "missing"), "--data-dir", str(tmp_path)]) == 1
    assert "Walk-forward evaluation failed" in capsys.readouterr().err
