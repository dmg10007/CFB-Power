import pandas as pd

from cfb_power.features import build_matchup_features, build_team_pregame_features


def source_frames():
    games = pd.DataFrame([
        {"game_id": "g1", "season": 2025, "week": 1, "game_date": "2025-09-01T18:00:00Z", "home_team_id": "A", "away_team_id": "B", "home_score": 30, "away_score": 20, "neutral_site": False, "completed": True},
        {"game_id": "g2", "season": 2025, "week": 2, "game_date": "2025-09-08T18:00:00Z", "home_team_id": "A", "away_team_id": "B", "home_score": 10, "away_score": 40, "neutral_site": False, "completed": True},
    ])
    rows = []
    for game_id, home_points, away_points in [("g1", 30, 20), ("g2", 10, 40)]:
        for team, opponent, is_home, points, allowed in [("A", "B", True, home_points, away_points), ("B", "A", False, away_points, home_points)]:
            rows.append({"game_id": game_id, "team_id": team, "opponent_id": opponent, "is_home": is_home, "points": points, "points_allowed": allowed, "total_yards": 300, "offensive_plays": 60, "passing_yards": 180, "rushing_yards": 120, "turnovers": 1, "takeaways": 1, "third_down_attempts": 10, "third_down_conversions": 5, "red_zone_attempts": 3, "red_zone_touchdowns": 2, "penalties": 5, "penalty_yards": 45, "time_of_possession_seconds": 1800})
    return games, pd.DataFrame(rows)


def test_rolling_features_use_only_prior_games():
    games, stats = source_frames()
    features = build_team_pregame_features(games, stats)
    team_a = features[features["team_id"] == "A"].sort_values("game_date")
    assert pd.isna(team_a.iloc[0]["points_l3"])
    assert team_a.iloc[1]["points_l3"] == 30


def test_matchups_get_home_and_away_feature_prefixes():
    games, stats = source_frames()
    team_features = build_team_pregame_features(games, stats)
    matchups = build_matchup_features(games, team_features)
    assert "home_points_l3" in matchups
    assert "away_points_l3" in matchups
