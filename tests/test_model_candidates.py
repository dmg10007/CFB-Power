"""Candidate model tests: scaling, shrinkage, and walk-forward integration; no API key needed."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cfb_power.dataset_workflow import DATASET_SCHEMA, DatasetError
from cfb_power.ingestion.cfbd_canonical import PLAY_COUNT_BASIS
from cfb_power.schemas import GAME_COLUMNS, TEAM_GAME_STAT_COLUMNS

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

from cfb_power.features import build_matchup_features, build_team_pregame_features
from cfb_power.model_candidates import (
    CANDIDATE_NAMES,
    SMALL_FEATURES,
    add_prior_game_counts,
    fit_scaled_ridge,
    predict_scaled,
    shrunk_expectation,
)
from cfb_power.validation import validate_games, validate_team_game_stats
from cfb_power.walk_forward import MODEL_NAMES, main, walk_forward_evaluate


def build_matchups(games, stats):
    games = validate_games(games)
    features = build_team_pregame_features(games, validate_team_game_stats(stats, games))
    return add_prior_game_counts(build_matchup_features(games, features), features)


def split(matchups, last_train_week=3, target_week=4):
    return matchups[matchups["week"] <= last_train_week], matchups[matchups["week"] == target_week]


def test_prior_game_counts_match_each_teams_history():
    games, stats = make_tables()
    matchups = build_matchups(games, stats)
    week4 = matchups[matchups["week"] == 4]
    assert set(week4["prior_games_home"]) == {3} and set(week4["prior_games_away"]) == {3}
    assert set(matchups[matchups["week"] == 1]["prior_games_home"]) == {0}
    assert not any(column.startswith(("home_", "away_")) and "prior" in column for column in matchups.columns)


def test_shrinkage_with_zero_prior_reproduces_the_rolling_average_baseline():
    games, stats = make_tables()
    train, target = split(build_matchups(games, stats))
    home, away, probability = shrunk_expectation(target, train, 0.0, False)
    expected_home = target[["home_points_season", "away_points_allowed_season"]].mean(axis=1)
    expected_away = target[["away_points_season", "home_points_allowed_season"]].mean(axis=1)
    pd.testing.assert_series_equal(home, expected_home, check_names=False)
    pd.testing.assert_series_equal(away, expected_away, check_names=False)
    assert probability.between(0, 1).all()


def test_large_prior_pulls_every_prediction_to_the_league_mean():
    games, stats = make_tables()
    train, target = split(build_matchups(games, stats))
    home, away, _ = shrunk_expectation(target, train, 1e9, False)
    league = pd.concat([train["home_score"], train["away_score"]]).mean()
    assert np.allclose(home, league, atol=1e-3) and np.allclose(away, league, atol=1e-3)


def test_shrinkage_keeps_teams_without_history_at_the_league_mean():
    games, stats = make_tables()
    matchups = build_matchups(games, stats)
    week1 = matchups[matchups["week"] == 1]
    train = matchups[matchups["week"] <= 3]
    home, away, _ = shrunk_expectation(week1, train, 2.0, False)
    league = pd.concat([train["home_score"], train["away_score"]]).mean()
    assert np.allclose(home, league) and np.allclose(away, league)


def test_home_advantage_shifts_margin_by_the_training_estimate_only_on_non_neutral_games():
    games, stats = make_tables()
    games.loc[games["week"] == 4, "neutral_site"] = True
    games.loc[games["game_id"] == 25, "neutral_site"] = False
    train, target = split(build_matchups(games, stats))
    plain_home, plain_away, _ = shrunk_expectation(target, train, 5.0, False)
    adjusted_home, adjusted_away, _ = shrunk_expectation(target, train, 5.0, True)
    shift = ((adjusted_home - adjusted_away) - (plain_home - plain_away))
    advantage = float((train["home_score"] - train["away_score"]).mean())
    non_neutral = target["home_field_indicator"] == 1
    assert non_neutral.sum() == 1
    assert np.allclose(shift[non_neutral], advantage)
    assert np.allclose(shift[~non_neutral], 0.0)


def test_scaled_ridge_is_invariant_to_feature_units():
    games, stats = make_tables()
    train, target = split(build_matchups(games, stats))
    baseline = predict_scaled(target, fit_scaled_ridge(train, SMALL_FEATURES))
    train_rescaled, target_rescaled = train.copy(), target.copy()
    for frame in (train_rescaled, target_rescaled):
        frame["home_yards_per_play_season"] *= 100
        frame["away_points_season"] *= 0.01
    rescaled = predict_scaled(target_rescaled, fit_scaled_ridge(train_rescaled, SMALL_FEATURES))
    assert np.allclose(baseline[0], rescaled[0]) and np.allclose(baseline[1], rescaled[1])
    assert baseline[2].between(0, 1).all()


def test_candidates_are_off_by_default(tmp_path):
    games, stats = make_tables()
    run = walk_forward_evaluate(write_dataset(tmp_path, games, stats), tmp_path)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["candidates"] == {"enabled": False, "definitions": {}}
    assert set(pd.read_csv(run / "overall_metrics.csv")["model"]) == set(MODEL_NAMES)


def test_candidates_are_scored_on_the_same_out_of_sample_games(tmp_path):
    games, stats = make_tables()
    run = walk_forward_evaluate(write_dataset(tmp_path, games, stats), tmp_path, candidates=True)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["candidates"]["enabled"] and set(manifest["candidates"]["definitions"]) == set(CANDIDATE_NAMES)
    assert manifest["counts"]["tested_games"] == 24
    assert any("chance" in note for note in manifest["notes"])
    overall = pd.read_csv(run / "overall_metrics.csv")
    assert list(overall["model"]) == [*MODEL_NAMES, *CANDIDATE_NAMES] and set(overall["games"]) == {24}
    weekly = pd.read_csv(run / "weekly_metrics.csv")
    assert len(weekly) == 3 * (len(MODEL_NAMES) + len(CANDIDATE_NAMES))
    predictions = pd.read_csv(run / "predictions.csv")
    for name in CANDIDATE_NAMES:
        assert predictions[[f"{name}_home_points", f"{name}_away_points"]].notna().all().all()
        assert predictions[f"{name}_home_win_probability"].between(0, 1).all()
    for metric in ("spread_mae", "total_mae", "brier_score"):
        assert f"{metric}_shrunk_k5_vs_rolling_average" in manifest["ridge_improvement_vs_baselines_pct"]
    base = walk_forward_evaluate(write_dataset(tmp_path, games, stats, "plain"), tmp_path)
    plain = pd.read_csv(base / "predictions.csv")
    shared = [column for column in plain.columns]
    pd.testing.assert_frame_equal(plain[shared], predictions[shared])


def test_later_results_never_change_any_candidate_prediction(tmp_path):
    games, stats = make_tables()
    first = pd.read_csv(walk_forward_evaluate(write_dataset(tmp_path, games, stats, "a"), tmp_path, candidates=True) / "predictions.csv")
    changed_games, changed_stats = shift_last_week(games, stats, 6)
    second = pd.read_csv(
        walk_forward_evaluate(write_dataset(tmp_path, changed_games, changed_stats, "b"), tmp_path, candidates=True) / "predictions.csv"
    )
    columns = [column for column in first.columns if column.startswith((*MODEL_NAMES, *CANDIDATE_NAMES))]
    pd.testing.assert_frame_equal(first[["game_id", *columns]], second[["game_id", *columns]])
    assert not first["home_score"].equals(second["home_score"])


def test_cli_flag_prints_candidate_rows(tmp_path, capsys):
    games, stats = make_tables()
    dataset = write_dataset(tmp_path, games, stats)
    assert main(["--dataset", str(dataset), "--data-dir", str(tmp_path), "--with-candidates"]) == 0
    output = capsys.readouterr().out
    assert all(name in output for name in CANDIDATE_NAMES)
