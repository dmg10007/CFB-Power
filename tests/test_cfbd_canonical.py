"""Offline canonical mapping tests using user-supplied CFBD responses."""

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from cfb_power.ingestion.cfbd_canonical import PLAY_COUNT_BASIS, normalize_cfbd_canonical
from cfb_power.schemas import GAME_COLUMNS, ROLLING_METRICS, TEAM_GAME_STAT_COLUMNS
from cfb_power.validation import DataValidationError

FIXTURES = Path(__file__).parent / "fixtures" / "cfbd"


@pytest.fixture()
def payloads():
    games = json.loads((FIXTURES / "canonical_games_401856766.json").read_text(encoding="utf-8"))
    stats = json.loads((FIXTURES / "canonical_team_stats_401856766.json").read_text(encoding="utf-8"))
    return games, stats


def set_stat(payload, category, value):
    for item in payload[0]["teams"][0]["stats"]:
        if item["category"] == category:
            item["stat"] = value
            return
    raise AssertionError(category)


def test_real_sample_mapping_and_neutral_site(payloads):
    games, stats = normalize_cfbd_canonical(*payloads)
    assert list(games.columns) == GAME_COLUMNS
    assert set(TEAM_GAME_STAT_COLUMNS).issubset(stats.columns)
    game = games.iloc[0]
    assert game["game_id"] == 401856766
    assert game["home_team_id"] == 2628 and game["away_team_id"] == 153
    assert bool(game["neutral_site"]) and bool(game["completed"])
    assert game["game_date"] == pd.Timestamp("2026-08-29T16:00:00Z")
    by_team = stats.set_index("team_id")
    tcu, unc = by_team.loc[2628], by_team.loc[153]
    assert tcu["opponent_id"] == 153 and unc["opponent_id"] == 2628
    assert bool(tcu["is_home"]) and not bool(unc["is_home"])
    assert (tcu["points"], tcu["points_allowed"]) == (10, 15)
    assert (unc["points"], unc["points_allowed"]) == (15, 10)
    assert (tcu["offensive_plays"], unc["offensive_plays"]) == (71, 57)
    assert (tcu["takeaways"], unc["takeaways"]) == (2, 1)
    assert (tcu["third_down_conversions"], tcu["third_down_attempts"]) == (7, 16)
    assert (unc["third_down_conversions"], unc["third_down_attempts"]) == (3, 11)
    assert (tcu["penalties"], tcu["penalty_yards"]) == (11, 90)
    assert (unc["penalties"], unc["penalty_yards"]) == (3, 20)
    assert (tcu["time_of_possession_seconds"], unc["time_of_possession_seconds"]) == (1982, 1618)
    assert stats["offensive_plays_basis"].eq(PLAY_COUNT_BASIS).all()
    assert stats[["red_zone_attempts", "red_zone_touchdowns"]].isna().all().all()
    assert "red_zone_td_rate" not in ROLLING_METRICS


def test_team_order_and_display_names_do_not_control_identity(payloads):
    games, boxes = payloads
    expected = normalize_cfbd_canonical(games, boxes)[1].sort_values("team_id").reset_index(drop=True)
    changed = copy.deepcopy(boxes)
    changed[0]["teams"].reverse()
    for team in changed[0]["teams"]:
        team["team"] = "Renamed program"
    actual = normalize_cfbd_canonical(games, changed)[1].sort_values("team_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)


def test_upcoming_fixture_has_no_final_stats(payloads):
    games, _ = payloads
    games[0].update(completed=False, homePoints=None, awayPoints=None)
    canonical, stats = normalize_cfbd_canonical(games, [])
    assert canonical[["home_score", "away_score"]].isna().all().all()
    assert stats.empty


def test_zero_turnovers_remain_zero(payloads):
    games, boxes = payloads
    set_stat(boxes, "turnovers", "0")
    stats = normalize_cfbd_canonical(games, boxes)[1].set_index("team_id")
    assert stats.loc[2628, "turnovers"] == 0
    assert stats.loc[153, "takeaways"] == 0


@pytest.mark.parametrize("category,value", [
    ("thirdDownEff", "7/16"), ("thirdDownEff", "17-16"),
    ("completionAttempts", "33-32"), ("possessionTime", "33:60"),
    ("possessionTime", "33:2"), ("turnovers", None),
    ("turnovers", "1.5"), ("rushingAttempts", "-1"),
    ("totalPenaltiesYards", "11/90"), ("totalYards", "280"),
])
def test_malformed_or_inconsistent_statistics_fail(payloads, category, value):
    games, boxes = payloads
    set_stat(boxes, category, value)
    with pytest.raises(DataValidationError):
        normalize_cfbd_canonical(games, boxes)


@pytest.mark.parametrize("case", [
    "duplicate_game", "duplicate_box", "unknown_game", "missing_box",
    "missing_opponent", "duplicate_team", "wrong_team", "wrong_role",
    "wrong_score", "duplicate_category", "missing_category", "bad_boolean",
    "bad_date", "tbd_date", "non_fbs", "unfinished_score", "unfinished_box",
])
def test_invalid_relationships_and_schedule_fail(payloads, case):
    games, boxes = payloads
    team = boxes[0]["teams"][0]
    if case == "duplicate_game":
        games.append(copy.deepcopy(games[0]))
    elif case == "duplicate_box":
        boxes.append(copy.deepcopy(boxes[0]))
    elif case == "unknown_game":
        boxes[0]["id"] = 999
    elif case == "missing_box":
        boxes.clear()
    elif case == "missing_opponent":
        boxes[0]["teams"].pop()
    elif case == "duplicate_team":
        boxes[0]["teams"][1]["teamId"] = team["teamId"]
    elif case == "wrong_team":
        team["teamId"] = 999
    elif case == "wrong_role":
        team["homeAway"] = "away"
    elif case == "wrong_score":
        team["points"] = 11
    elif case == "duplicate_category":
        team["stats"].append(copy.deepcopy(team["stats"][0]))
    elif case == "missing_category":
        team["stats"] = [item for item in team["stats"] if item["category"] != "turnovers"]
    elif case == "bad_boolean":
        games[0]["completed"] = "false"
    elif case == "bad_date":
        games[0]["startDate"] = "not-a-date"
    elif case == "tbd_date":
        games[0]["startTimeTBD"] = True
    elif case == "non_fbs":
        games[0]["awayClassification"] = "fcs"
    elif case == "unfinished_score":
        games[0]["completed"] = False
    elif case == "unfinished_box":
        games[0].update(completed=False, homePoints=None, awayPoints=None)
    with pytest.raises(DataValidationError):
        normalize_cfbd_canonical(games, boxes)


def test_outputs_pass_existing_canonical_validation(payloads):
    from cfb_power.validation import validate_games, validate_team_game_stats

    games, stats = normalize_cfbd_canonical(*payloads)
    checked_games = validate_games(games)
    checked_stats = validate_team_game_stats(stats, checked_games)
    assert len(checked_games) == 1 and len(checked_stats) == 2


def test_prior_game_features_work_with_missing_red_zone(payloads):
    from cfb_power.features import build_team_pregame_features

    games, stats = normalize_cfbd_canonical(*payloads)
    later_games = games.copy()
    later_games["game_id"] = 401856767
    later_games["week"] = 2
    later_games["game_date"] += pd.Timedelta(days=7)
    later_stats = stats.copy()
    later_stats["game_id"] = 401856767
    later_stats["points"] = 99
    features = build_team_pregame_features(
        pd.concat([games, later_games], ignore_index=True),
        pd.concat([stats, later_stats], ignore_index=True),
    )
    later = features[features["game_id"] == 401856767].set_index("team_id")
    assert later.loc[2628, "points_l3"] == 10
    assert later.loc[153, "points_l3"] == 15
    assert not any(column.startswith("red_zone_td_rate_") for column in features)
