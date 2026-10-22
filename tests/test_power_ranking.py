"""Tests for CFB Power Rankings, statistical efficiency metrics, and team comparison."""

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from cfb_power.cli import app
from cfb_power.power_ranking import (
    DEFAULT_HOME_ADVANTAGE,
    PowerRankings,
    fit_power_rankings,
)
from cfb_power.schemas import PREDICTION_COLUMNS

runner = CliRunner()


def make_sample_season_data(teams=6, weeks=8, seed=42):
    rng = np.random.default_rng(seed)
    team_ids = [f"TEAM_{i:02d}" for i in range(1, teams + 1)]
    # True latent strengths: TEAM_01 is strongest (+10), TEAM_06 is weakest (-10)
    strengths = {team_ids[i]: 10.0 - (20.0 / (teams - 1)) * i for i in range(teams)}

    game_rows = []
    stat_rows = []
    gid = 1

    for week in range(1, weeks + 1):
        perm = list(rng.permutation(team_ids))
        for idx in range(0, len(perm) - 1, 2):
            away, home = perm[idx], perm[idx + 1]
            h_score = max(
                3, int(round(28 + strengths[home] - 0.3 * strengths[away] + 2.5 + rng.normal(0, 5)))
            )
            a_score = max(
                3, int(round(24 + strengths[away] - 0.3 * strengths[home] + rng.normal(0, 5)))
            )

            game_rows.append(
                {
                    "game_id": gid,
                    "season": 2026,
                    "week": week,
                    "game_date": f"2026-09-{min(28, week * 3):02d}T18:00:00Z",
                    "home_team_id": home,
                    "away_team_id": away,
                    "home_score": h_score,
                    "away_score": a_score,
                    "neutral_site": False,
                    "completed": True,
                }
            )

            for team, opp, is_h, pts, opp_pts in [
                (home, away, True, h_score, a_score),
                (away, home, False, a_score, h_score),
            ]:
                plays = int(rng.integers(60, 75))
                yds = max(180, int(pts * 12 + strengths[team] * 6 + rng.normal(0, 30)))
                stat_rows.append(
                    {
                        "game_id": gid,
                        "team_id": team,
                        "opponent_id": opp,
                        "is_home": is_h,
                        "points": pts,
                        "points_allowed": opp_pts,
                        "total_yards": yds,
                        "offensive_plays": plays,
                        "passing_yards": int(yds * 0.6),
                        "rushing_yards": int(yds * 0.4),
                        "turnovers": int(rng.integers(0, 3)),
                        "takeaways": int(rng.integers(0, 3)),
                        "third_down_attempts": 12,
                        "third_down_conversions": int(rng.integers(3, 8)),
                        "red_zone_attempts": 4,
                        "red_zone_touchdowns": int(rng.integers(1, 4)),
                        "penalties": 5,
                        "penalty_yards": 45,
                        "time_of_possession_seconds": 1800,
                    }
                )
            gid += 1

    return pd.DataFrame(game_rows), pd.DataFrame(stat_rows)


def test_fit_power_rankings_ordering_and_components():
    games, stats = make_sample_season_data()
    rankings = fit_power_rankings(games, stats)

    assert isinstance(rankings, PowerRankings)
    df = rankings.to_dataframe()
    assert len(df) == 6
    assert list(df["rank"]) == [1, 2, 3, 4, 5, 6]

    # Strongest team (TEAM_01) should rank higher than weakest team (TEAM_06)
    team_01 = rankings.get_team("TEAM_01")
    team_06 = rankings.get_team("TEAM_06")
    assert team_01.rank < team_06.rank
    assert team_01.power_rating > team_06.power_rating

    # Net power rating should equal offensive + defensive rating
    for profile in rankings.teams.values():
        assert profile.power_rating == pytest.approx(
            profile.offensive_rating + profile.defensive_rating, abs=0.01
        )
        assert profile.yards_per_play > 0
        assert 0.0 <= profile.third_down_rate <= 1.0
        assert 0.0 <= profile.red_zone_td_rate <= 1.0


def test_compare_teams_head_to_head():
    games, stats = make_sample_season_data()
    rankings = fit_power_rankings(games, stats)

    # Neutral site matchup
    comp_neutral = rankings.compare_teams("TEAM_01", "TEAM_06", neutral_site=True)
    assert comp_neutral.team_a.team_id == "TEAM_01"
    assert comp_neutral.team_b.team_id == "TEAM_06"
    assert comp_neutral.neutral_site is True
    assert comp_neutral.projected_spread > 0  # TEAM_01 favored
    assert comp_neutral.team_a_win_probability > 0.5
    assert (
        comp_neutral.team_a_win_probability + comp_neutral.team_b_win_probability
        == pytest.approx(1.0)
    )
    assert comp_neutral.projected_total == pytest.approx(
        comp_neutral.team_a_expected_points + comp_neutral.team_b_expected_points
    )

    # Home matchup for Team A
    comp_home_a = rankings.compare_teams(
        "TEAM_01", "TEAM_06", neutral_site=False, home_team="TEAM_01"
    )
    # Spread should favor Team A more when playing at home by exactly home_advantage
    assert comp_home_a.projected_spread == pytest.approx(
        comp_neutral.projected_spread + DEFAULT_HOME_ADVANTAGE, abs=0.2
    )

    # Stat comparison table
    assert not comp_neutral.stat_comparison.empty
    assert "Metric" in comp_neutral.stat_comparison.columns


def test_roster_and_qb_adjustments():
    games, stats = make_sample_season_data()
    rankings = fit_power_rankings(games, stats)

    base = rankings.compare_teams("TEAM_01", "TEAM_02", neutral_site=True)

    # Apply penalty for Team A starting backup QB (-4.0 pts)
    adjusted = rankings.compare_teams(
        "TEAM_01",
        "TEAM_02",
        neutral_site=True,
        roster_adjustments={"TEAM_01": -4.0},
    )

    assert adjusted.team_a_expected_points == pytest.approx(
        base.team_a_expected_points - 4.0, abs=0.1
    )
    assert adjusted.projected_spread == pytest.approx(base.projected_spread - 4.0, abs=0.1)
    assert adjusted.team_a_win_probability < base.team_a_win_probability


def test_predict_slate_schema_compliance():
    games, stats = make_sample_season_data()
    rankings = fit_power_rankings(games, stats)

    # Predict upcoming slate of fixtures
    fixtures = pd.DataFrame(
        [
            {
                "game_id": 901,
                "season": 2026,
                "week": 9,
                "game_date": "2026-10-31T19:00:00Z",
                "home_team_id": "TEAM_01",
                "away_team_id": "TEAM_02",
                "neutral_site": False,
            },
            {
                "game_id": 902,
                "season": 2026,
                "week": 9,
                "game_date": "2026-10-31T22:30:00Z",
                "home_team_id": "TEAM_03",
                "away_team_id": "TEAM_04",
                "neutral_site": True,
            },
        ]
    )

    preds = rankings.predict_slate(fixtures)
    assert list(preds.columns) == PREDICTION_COLUMNS
    assert len(preds) == 2
    assert preds["home_expected_points"].iloc[0] > 0
    assert 0.0 < preds["home_win_probability"].iloc[0] < 1.0


def test_cli_power_rankings_and_compare(tmp_path, monkeypatch):
    monkeypatch.setenv("CFB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CFB_EXPORT_DIR", str(tmp_path / "exports"))
    monkeypatch.setenv("CFB_MODEL_DIR", str(tmp_path / "models"))

    # 1. Create synthetic data
    runner.invoke(app, ["create-demo-data", "--seasons", "1", "--teams", "6", "--weeks", "6"])

    # 2. Test power-rankings CLI command
    rank_res = runner.invoke(app, ["power-rankings", "--top", "5"])
    assert rank_res.exit_code == 0, rank_res.output
    assert "CFB Power Rankings" in rank_res.output
    assert "TEAM_01" in rank_res.output or "Rank" in rank_res.output

    # 3. Test compare CLI command
    comp_res = runner.invoke(app, ["compare", "TEAM_01", "TEAM_02", "--neutral"])
    assert comp_res.exit_code == 0, comp_res.output
    assert "Matchup Forecast" in comp_res.output
    assert "Tale of the Tape" in comp_res.output
    assert "TEAM_01" in comp_res.output
    assert "TEAM_02" in comp_res.output


def test_compare_teams_with_individual_adjustments():
    games, stats = make_sample_season_data(teams=4, weeks=4)
    rankings = fit_power_rankings(games, stats)

    # Base matchup
    base = rankings.compare_teams("TEAM_01", "TEAM_02", neutral_site=True)

    # Matchup with team_a_adj and team_b_adj
    adj = rankings.compare_teams(
        "TEAM_01", "TEAM_02", neutral_site=True, team_a_adj=3.5, team_b_adj=-2.0
    )

    assert adj.team_a_expected_points > base.team_a_expected_points
    assert adj.team_b_expected_points < base.team_b_expected_points
    assert adj.projected_spread > base.projected_spread
    assert base.team_a_profile.time_of_possession_min > 0.0
    assert base.team_b_profile.time_of_possession_min > 0.0
