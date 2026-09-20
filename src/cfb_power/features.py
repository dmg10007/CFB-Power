from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import ROLLING_METRICS
from .validation import validate_games, validate_no_future_feature_data, validate_team_game_stats


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.divide(denominator.replace(0, np.nan))


def _add_rate_metrics(stats: pd.DataFrame) -> pd.DataFrame:
    result = stats.copy()
    result["yards_per_play"] = _safe_divide(result["total_yards"], result["offensive_plays"])
    result["passing_yards_per_play"] = _safe_divide(result["passing_yards"], result["offensive_plays"])
    result["rushing_yards_per_play"] = _safe_divide(result["rushing_yards"], result["offensive_plays"])
    result["third_down_rate"] = _safe_divide(
        result["third_down_conversions"], result["third_down_attempts"]
    )
    result["red_zone_td_rate"] = _safe_divide(
        result["red_zone_touchdowns"], result["red_zone_attempts"]
    )
    return result


def build_team_pregame_features(games: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Build pregame rolling team features; every aggregation is shifted one game."""
    games = validate_games(games)
    stats = validate_team_game_stats(stats, games)
    context = games[["game_id", "season", "week", "game_date", "completed"]]
    team_games = _add_rate_metrics(stats).merge(context, on="game_id", how="inner", validate="many_to_one")
    team_games = team_games.sort_values(["team_id", "game_date", "game_id"]).copy()

    for metric in ROLLING_METRICS:
        grouped = team_games.groupby(["team_id", "season"], group_keys=False)[metric]
        history = grouped.shift(1)
        team_games[f"{metric}_l3"] = history.groupby(
            [team_games["team_id"], team_games["season"]]
        ).transform(lambda series: series.rolling(3, min_periods=1).mean())
        team_games[f"{metric}_l5"] = history.groupby(
            [team_games["team_id"], team_games["season"]]
        ).transform(lambda series: series.rolling(5, min_periods=1).mean())
        team_games[f"{metric}_season"] = history.groupby(
            [team_games["team_id"], team_games["season"]]
        ).transform(lambda series: series.expanding(min_periods=1).mean())

    validate_no_future_feature_data(team_games.dropna(subset=["points_l3"], how="all"))
    return team_games


def build_matchup_features(games: pd.DataFrame, team_features: pd.DataFrame) -> pd.DataFrame:
    """Create one pregame row per fixture with home and away team feature prefixes."""
    games = validate_games(games)
    feature_columns = [
        column
        for column in team_features.columns
        if column.endswith(("_l3", "_l5", "_season"))
    ]
    home = team_features[["game_id", "team_id", *feature_columns]].copy()
    away = team_features[["game_id", "team_id", *feature_columns]].copy()
    home = home.rename(columns={"team_id": "home_team_id", **{c: f"home_{c}" for c in feature_columns}})
    away = away.rename(columns={"team_id": "away_team_id", **{c: f"away_{c}" for c in feature_columns}})
    matchups = games.merge(home, on=["game_id", "home_team_id"], how="left")
    matchups = matchups.merge(away, on=["game_id", "away_team_id"], how="left")
    matchups["home_field_indicator"] = (~matchups["neutral_site"].astype(bool)).astype(int)
    matchups["home_target_points"] = matchups["home_score"]
    matchups["away_target_points"] = matchups["away_score"]
    return matchups


def model_feature_columns(matchups: pd.DataFrame) -> list[str]:
    """Return unique numeric predictor columns in deterministic order."""
    excluded = {
        "home_team_id",
        "away_team_id",
        "home_score",
        "away_score",
        "home_target_points",
        "away_target_points",
    }
    feature_columns = [
        column
        for column in matchups.columns
        if column.startswith(("home_", "away_"))
        and column not in excluded
        and pd.api.types.is_numeric_dtype(matchups[column])
    ]
    return list(dict.fromkeys([*feature_columns, "home_field_indicator"]))
