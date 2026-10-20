import numpy as np
import pandas as pd
import pytest

from cfb_power.power_ratings import RatingSettings, build_power_ratings
from cfb_power.rating_workflow import audit_source_coverage

CUTOFF = "2026-10-19T12:00:00Z"


def game(identifier=1, home=1, away=2, hs=35, aws=21, date="2026-10-01T12:00:00Z",
         season=2026, neutral=True, completed=True):
    return {"game_id": identifier, "season": season, "week": 1, "game_date": date,
            "home_team_id": home, "away_team_id": away, "home_score": hs,
            "away_score": aws, "neutral_site": neutral, "completed": completed}


def schedule():
    rows = []
    offense, defense = {1: 8, 2: 2, 3: -3, 4: -7}, {1: 5, 2: 2, 3: -2, 4: -5}
    for i, home in enumerate(offense):
        for away in offense:
            if home != away:
                rows.append(game(len(rows)+1, home, away,
                                 30+offense[home]-defense[away]+2,
                                 30+offense[away]-defense[home]-2,
                                 date=f"2026-10-{1+i:02d}T12:00:00Z", neutral=False))
    return pd.DataFrame(rows)


def snapshot(games=None, **kwargs):
    return build_power_ratings(schedule() if games is None else games,
                               season=2026, as_of=CUTOFF, **kwargs)


def test_centered_ratings_and_power_identity():
    result = snapshot()
    assert result.teams.offense_rating.mean() == pytest.approx(0, abs=1e-10)
    assert result.teams.defense_rating.mean() == pytest.approx(0, abs=1e-10)
    np.testing.assert_allclose(result.teams.power_rating,
                               result.teams.offense_rating + result.teams.defense_rating)
    assert result.teams.team_id.tolist() == [1, 2, 3, 4]


def test_neutral_margin_is_rating_difference():
    result = snapshot()
    table = result.teams.set_index("team_id")
    prediction = result.compare(1, 2)
    assert prediction["margin_a"] == pytest.approx(table.loc[1,"power_rating"] - table.loc[2,"power_rating"])
    assert prediction["total"] == pytest.approx(prediction["team_a_expected_points"] + prediction["team_b_expected_points"])


def test_swapping_teams_swaps_scores_on_neutral_field():
    result = snapshot()
    a, b = result.compare(1, 2), result.compare(2, 1)
    assert a["margin_a"] == pytest.approx(-b["margin_a"])
    assert a["team_a_expected_points"] == pytest.approx(b["team_b_expected_points"])


def test_learned_venue_is_applied_once():
    result = snapshot()
    neutral = result.compare(1, 2)
    home, away = result.compare(1, 2, "team_a_home"), result.compare(1, 2, "team_b_home")
    assert result.home_field_points > 0
    assert home["margin_a"] - neutral["margin_a"] == pytest.approx(result.home_field_points)
    assert away["margin_a"] - neutral["margin_a"] == pytest.approx(-result.home_field_points)
    assert home["total"] == pytest.approx(neutral["total"])


def test_neutral_training_cannot_learn_venue_effect():
    games = schedule().assign(neutral_site=True)
    assert snapshot(games).home_field_points == pytest.approx(0)


def test_future_outcomes_do_not_change_ratings():
    games = schedule()
    later = game(99, 5, 6, 80, 0, "2026-10-20T12:00:00Z")
    result = snapshot(pd.concat([games, pd.DataFrame([later])], ignore_index=True))
    pd.testing.assert_frame_equal(result.teams, snapshot(games).teams)
    assert 99 not in result.metadata["training_game_ids"]


def test_kickoff_at_cutoff_is_excluded():
    games = pd.DataFrame([game(), game(2, hs=90, date=CUTOFF)])
    assert snapshot(games).metadata["current_games"] == 1


def test_result_availability_uses_strict_cutoff():
    games = pd.DataFrame([game(), game(2, hs=90)])
    games["result_available_at_utc"] = ["2026-10-02T12:00:00Z", CUTOFF]
    assert snapshot(games).metadata["training_game_ids"] == [1]


def test_missing_result_availability_is_not_assumed_known():
    games = pd.DataFrame([game(), game(2, hs=90)])
    games["result_available_at_utc"] = ["2026-10-02T12:00:00Z", None]
    assert snapshot(games).metadata["training_game_ids"] == [1]


def test_absent_knowledge_times_warn():
    assert any("knowledge times" in x for x in snapshot().metadata["warnings"])


def test_prior_season_changes_early_season_estimates():
    current = pd.DataFrame([game()])
    previous = game(20, hs=60, aws=10, season=2025, date="2025-11-01T12:00:00Z")
    all_games = pd.concat([current, pd.DataFrame([previous])], ignore_index=True)
    with_prior = snapshot(all_games)
    without_prior = snapshot(all_games, settings=RatingSettings(previous_season_carry=0))
    assert with_prior.compare(1,2)["margin_a"] > without_prior.compare(1,2)["margin_a"]
    assert with_prior.teams.previous_season_games.sum() == 2


def test_preseason_prior_only_is_labeled_and_venue_is_zero():
    games = pd.DataFrame([game(season=2025, date="2025-11-01T12:00:00Z")])
    result = snapshot(games)
    assert (result.teams.games_played == 0).all()
    assert result.home_field_points == 0
    assert any("discounted previous" in x for x in result.metadata["warnings"])


def test_older_seasons_do_not_change_current_ratings():
    old = game(99, hs=100, aws=0, season=2024, date="2024-11-01T12:00:00Z")
    result = snapshot(pd.concat([schedule(), pd.DataFrame([old])], ignore_index=True))
    pd.testing.assert_frame_equal(result.teams, snapshot().teams)


def test_recency_weights_favor_recent_performance():
    games = pd.DataFrame([game(1, hs=50, aws=10, date="2026-09-01T12:00:00Z"),
                          game(2, hs=10, aws=50, date="2026-10-18T12:00:00Z")])
    assert snapshot(games, settings=RatingSettings(half_life_days=7)).compare(1,2)["margin_a"] < 0


def test_explicit_unplayed_team_is_provisional():
    result = snapshot(team_ids=[1,2,3,4,5])
    row = result.teams.set_index("team_id").loc[5]
    assert row.games_played == 0 and row.low_history
    assert row.prior_source == "league_average"


def test_disconnected_schedule_is_disclosed():
    result = snapshot(pd.DataFrame([game(), game(2,3,4)]))
    assert any("Disconnected" in x for x in result.metadata["warnings"])
    assert any("disconnected" in x for x in result.compare(1,3)["warnings"])


def test_unfinished_game_is_not_training_data():
    games = pd.DataFrame([game(), game(2,hs=None,aws=None,completed=False)])
    result = snapshot(games)
    assert result.metadata["current_games"] == 1
    assert any("past-kickoff" in x for x in result.metadata["warnings"])


@pytest.mark.parametrize("settings", [dict(half_life_days=0),dict(team_penalty=-1),
    dict(venue_penalty=float("nan")),dict(previous_season_carry=2),dict(low_history_games=-1)])
def test_bad_settings_rejected(settings):
    with pytest.raises(ValueError):
        RatingSettings(**settings)


@pytest.mark.parametrize("mutation", ["duplicate", "negative", "self", "naive", "availability"])
def test_invalid_games_rejected(mutation):
    games = pd.DataFrame([game()])
    if mutation == "duplicate":
        games = pd.concat([games,games],ignore_index=True)
    elif mutation == "negative":
        games.loc[0,"home_score"] = -1
    elif mutation == "self":
        games.loc[0,"away_team_id"] = 1
    elif mutation == "naive":
        games.loc[0,"game_date"] = "2026-10-01"
    else:
        games["result_available_at_utc"] = "2026-09-01T12:00:00Z"
    with pytest.raises(ValueError):
        snapshot(games)


@pytest.mark.parametrize("a,b,venue", [(1,1,"neutral"),(1,99,"neutral"),(1,2,"invalid")])
def test_bad_matchups_rejected(a,b,venue):
    with pytest.raises(ValueError):
        snapshot().compare(a,b,venue)


def test_empty_history_rejected():
    games = pd.DataFrame([game(date="2026-10-20T12:00:00Z")])
    with pytest.raises(ValueError,match="No eligible"):
        snapshot(games)


def test_negative_linear_scores_are_not_silently_clipped():
    result = snapshot()
    result.base_points = -100
    with pytest.raises(ValueError, match="Negative linear"):
        result.compare(1,2)


def test_tied_rank_uses_same_rank_with_deterministic_order():
    result = snapshot(pd.DataFrame([game(hs=21,aws=21)]))
    assert result.teams.power_rank.tolist() == [1,1]
    assert result.teams.team_id.tolist() == [1,2]


def test_audit_missing_rates_not_filled_with_zero():
    games = pd.DataFrame([game()])
    stats = pd.DataFrame({"game_id":[1,1],"team_id":[1,2],
                          "third_down_attempts":[0,10],"third_down_conversions":[0,4]})
    audit = audit_source_coverage(games,stats,season=2026,as_of=CUTOFF)
    fields = pd.DataFrame(audit["features"]).set_index("feature")
    assert fields.loc["red_zone_td_rate","available_rows"] == 0
    assert fields.loc["third_down_conversion_rate","available_rows"] == 1
    assert fields.loc["injuries","status"] == "missing"
    assert audit["external_api_requests"] == 0


def test_audit_counts_missing_stat_rows_against_expected_games():
    games = pd.DataFrame([game()])
    stats = pd.DataFrame({"game_id":[1],"team_id":[1],"points":[35]})
    audit = audit_source_coverage(games,stats,season=2026,as_of=CUTOFF)
    points = next(x for x in audit["features"] if x["feature"]=="points")
    assert points["eligible_team_games"] == 2 and points["available_rows"] == 1


def test_opponent_adjustment_explains_different_schedule_results():
    games = pd.DataFrame([game(1,1,3,40,20),game(2,2,4,40,20),game(3,3,4,50,10)])
    result = snapshot(games)
    powers = result.teams.set_index("team_id").power_rating
    assert powers[1] > powers[2]


def test_deterministic_input_row_order():
    pd.testing.assert_frame_equal(snapshot(schedule()).teams,
                                  snapshot(schedule().sample(frac=1,random_state=3)).teams)
