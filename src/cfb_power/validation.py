from __future__ import annotations

import pandas as pd

from .schemas import GAME_COLUMNS, TEAM_GAME_STAT_COLUMNS


class DataValidationError(ValueError):
    """Raised when normalized source data violates the canonical contract."""


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise DataValidationError(f"{label} is missing required columns: {', '.join(missing)}")


def validate_games(games: pd.DataFrame) -> pd.DataFrame:
    _require_columns(games, GAME_COLUMNS, "games")
    checked = games.copy()
    checked["game_date"] = pd.to_datetime(checked["game_date"], errors="raise", utc=True)
    if checked["game_id"].duplicated().any():
        raise DataValidationError("games contains duplicate game_id values")
    if (checked[["home_score", "away_score"]].dropna() < 0).any().any():
        raise DataValidationError("games contains a negative final score")
    if (checked["home_team_id"] == checked["away_team_id"]).any():
        raise DataValidationError("games contains a team playing itself")
    return checked


def validate_team_game_stats(stats: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    _require_columns(stats, TEAM_GAME_STAT_COLUMNS, "team_game_stats")
    checked = stats.copy()
    if checked.duplicated(["game_id", "team_id"]).any():
        raise DataValidationError("team_game_stats contains duplicate game_id/team_id records")
    unknown_games = set(checked["game_id"]) - set(games["game_id"])
    if unknown_games:
        raise DataValidationError("team_game_stats references game_ids absent from games")
    if (checked["offensive_plays"] <= 0).any():
        raise DataValidationError("team_game_stats contains non-positive offensive_plays")
    return checked


def validate_no_future_feature_data(features: pd.DataFrame) -> None:
    feature_columns = [column for column in features.columns if column.endswith(("_l3", "_l5", "_season"))]
    if not feature_columns:
        raise DataValidationError("feature table has no rolling pregame columns")
    if features[feature_columns].isna().all(axis=None):
        raise DataValidationError("all rolling feature values are missing")
