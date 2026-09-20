from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from . import MODEL_VERSION
from .config import Paths
from .features import model_feature_columns


@dataclass
class ScoreModels:
    home: Pipeline
    away: Pipeline
    features: list[str]
    margin_residual_std: float


def _pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("regression", Ridge(alpha=12.0)),
        ]
    )


def fit_score_models(training_matchups: pd.DataFrame, paths: Paths | None = None) -> ScoreModels:
    train = training_matchups.dropna(subset=["home_target_points", "away_target_points"]).copy()
    features = model_feature_columns(train)
    if len(train) < 20:
        raise ValueError("At least 20 completed matchups are required to train V1.")
    home_model, away_model = _pipeline(), _pipeline()
    home_model.fit(train[features], train["home_target_points"])
    away_model.fit(train[features], train["away_target_points"])
    margin_residual = (train["home_target_points"] - train["away_target_points"]) - (
        home_model.predict(train[features]) - away_model.predict(train[features])
    )
    models = ScoreModels(
        home=home_model,
        away=away_model,
        features=features,
        margin_residual_std=max(float(np.std(margin_residual, ddof=1)), 1.0),
    )
    if paths:
        paths.ensure()
        dump(models, paths.model_file)
    return models


def predict_matchups(matchups: pd.DataFrame, models: ScoreModels) -> pd.DataFrame:
    result = matchups.copy()
    result["home_expected_points"] = models.home.predict(result[models.features]).clip(0, 70)
    result["away_expected_points"] = models.away.predict(result[models.features]).clip(0, 70)
    result["projected_spread_home"] = result["home_expected_points"] - result["away_expected_points"]
    result["projected_total"] = result["home_expected_points"] + result["away_expected_points"]
    z = result["projected_spread_home"] / models.margin_residual_std
    result["home_win_probability"] = 0.5 * (1.0 + np.vectorize(np.math.erf)(z / np.sqrt(2.0)))
    point_interval = models.margin_residual_std / np.sqrt(2.0)
    result["home_score_low"] = (result["home_expected_points"] - point_interval).clip(0).round(1)
    result["home_score_high"] = (result["home_expected_points"] + point_interval).round(1)
    result["away_score_low"] = (result["away_expected_points"] - point_interval).clip(0).round(1)
    result["away_score_high"] = (result["away_expected_points"] + point_interval).round(1)
    result["confidence_tier"] = pd.cut(
        result["home_win_probability"].sub(0.5).abs() * 2,
        bins=[-0.01, 0.16, 0.34, 1.0],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    result["model_version"] = MODEL_VERSION
    return result
