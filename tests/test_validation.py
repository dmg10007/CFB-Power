import pandas as pd
import pytest

from cfb_power.validation import DataValidationError, validate_games


def test_games_reject_duplicate_ids():
    games = pd.DataFrame([
        {"game_id": "g1", "season": 2025, "week": 1, "game_date": "2025-09-01T18:00:00Z", "home_team_id": "A", "away_team_id": "B", "home_score": 10, "away_score": 7, "neutral_site": False, "completed": True},
        {"game_id": "g1", "season": 2025, "week": 2, "game_date": "2025-09-08T18:00:00Z", "home_team_id": "A", "away_team_id": "C", "home_score": 10, "away_score": 7, "neutral_site": False, "completed": True},
    ])
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_games(games)


def test_games_reject_negative_scores():
    games = pd.DataFrame([
        {"game_id": "g1", "season": 2025, "week": 1, "game_date": "2025-09-01T18:00:00Z", "home_team_id": "A", "away_team_id": "B", "home_score": -1, "away_score": 7, "neutral_site": False, "completed": True},
    ])
    with pytest.raises(DataValidationError, match="negative"):
        validate_games(games)
