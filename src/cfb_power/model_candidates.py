"""Candidate score models for walk-forward comparison; not used for production forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import model_feature_columns
from .model import _normal_cdf

SMALL_FEATURES = (
    "home_points_season", "away_points_season",
    "home_points_allowed_season", "away_points_allowed_season",
    "home_yards_per_play_season", "away_yards_per_play_season",
    "home_field_indicator",
)
SCALED_ALPHA = 10.0
CANDIDATE_DEFINITIONS = {
    "ridge_scaled_all": f"Ridge (alpha={SCALED_ALPHA}) on standardized versions of every existing feature",
    "ridge_scaled_small": f"Ridge (alpha={SCALED_ALPHA}) on standardized season points, points allowed, yards per play, and home field",
    "shrunk_k2": "Season points/points-allowed averages shrunk toward the league mean with prior weight 2 games",
    "shrunk_k5": "Season points/points-allowed averages shrunk toward the league mean with prior weight 5 games",
    "shrunk_k5_home": "shrunk_k5 plus a training-estimated home-field adjustment on non-neutral games",
}


def add_prior_game_counts(matchups: pd.DataFrame, team_features: pd.DataFrame) -> pd.DataFrame:
    """Add prior_games_home/away: games each team had played earlier in its season."""
    ordered = team_features.sort_values(["team_id", "game_date", "game_id"])
    counts = ordered[["game_id", "team_id"]].copy()
    counts["prior_games"] = ordered.groupby(["team_id", "season"]).cumcount().to_numpy()
    home = counts.rename(columns={"team_id": "home_team_id", "prior_games": "prior_games_home"})
    away = counts.rename(columns={"team_id": "away_team_id", "prior_games": "prior_games_away"})
    result = matchups.merge(home, on=["game_id", "home_team_id"], how="left")
    result = result.merge(away, on=["game_id", "away_team_id"], how="left")
    for column in ("prior_games_home", "prior_games_away"):
        result[column] = result[column].fillna(0).astype(int)
    return result


@dataclass
class ScaledRidge:
    home: Pipeline
    away: Pipeline
    features: list
    residual_std: float


def _scaled_pipeline(alpha: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regression", Ridge(alpha=alpha)),
    ])


def fit_scaled_ridge(train: pd.DataFrame, features, alpha: float = SCALED_ALPHA) -> ScaledRidge:
    fit = train.dropna(subset=["home_target_points", "away_target_points"])
    features = list(features)
    home, away = _scaled_pipeline(alpha), _scaled_pipeline(alpha)
    home.fit(fit[features], fit["home_target_points"])
    away.fit(fit[features], fit["away_target_points"])
    residual = (fit["home_target_points"] - fit["away_target_points"]) - (
        home.predict(fit[features]) - away.predict(fit[features])
    )
    return ScaledRidge(home, away, features, max(float(np.std(residual, ddof=1)), 1.0))


def predict_scaled(target: pd.DataFrame, model: ScaledRidge):
    home = pd.Series(model.home.predict(target[model.features]), index=target.index).clip(0, 70)
    away = pd.Series(model.away.predict(target[model.features]), index=target.index).clip(0, 70)
    return home, away, pd.Series(_normal_cdf((home - away) / model.residual_std), index=target.index)


def _shrink(mean: pd.Series, games: pd.Series, prior_games: float, league: float) -> pd.Series:
    denominator = games + prior_games
    shrunk = (mean.fillna(0.0) * games + prior_games * league) / denominator.where(denominator > 0, np.nan)
    return shrunk.fillna(league)


def shrunk_expectation(target: pd.DataFrame, train: pd.DataFrame, prior_games: float, home_advantage: bool):
    """Average each side's shrunk offense with the opponent's shrunk defense.

    prior_games=0 reproduces the rolling-average baseline for teams with history.
    """
    league = float(pd.concat([train["home_score"], train["away_score"]]).mean())
    games_home, games_away = target["prior_games_home"], target["prior_games_away"]
    offense_home = _shrink(target["home_points_season"], games_home, prior_games, league)
    defense_home = _shrink(target["home_points_allowed_season"], games_home, prior_games, league)
    offense_away = _shrink(target["away_points_season"], games_away, prior_games, league)
    defense_away = _shrink(target["away_points_allowed_season"], games_away, prior_games, league)
    home = (offense_home + defense_away) / 2
    away = (offense_away + defense_home) / 2
    if home_advantage:
        home_games = train[train["home_field_indicator"] == 1]
        advantage = float((home_games["home_score"] - home_games["away_score"]).mean()) / 2 if len(home_games) else 0.0
        home = home + advantage * target["home_field_indicator"]
        away = away - advantage * target["home_field_indicator"]
    scale = max(float((train["home_score"] - train["away_score"]).std(ddof=1)), 1.0)
    return home, away, pd.Series(_normal_cdf((home - away) / scale), index=target.index)


def _scaled_all(train, target):
    return predict_scaled(target, fit_scaled_ridge(train, model_feature_columns(train)))


def _scaled_small(train, target):
    return predict_scaled(target, fit_scaled_ridge(train, SMALL_FEATURES))


CANDIDATES = {
    "ridge_scaled_all": _scaled_all,
    "ridge_scaled_small": _scaled_small,
    "shrunk_k2": lambda train, target: shrunk_expectation(target, train, 2.0, False),
    "shrunk_k5": lambda train, target: shrunk_expectation(target, train, 5.0, False),
    "shrunk_k5_home": lambda train, target: shrunk_expectation(target, train, 5.0, True),
}
CANDIDATE_NAMES = tuple(CANDIDATES)
