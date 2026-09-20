import numpy as np
import pandas as pd

from cfb_power.features import model_feature_columns
from cfb_power.model import _normal_cdf, fit_score_models, predict_matchups


def test_score_model_produces_required_outputs():
    rng = np.random.default_rng(9)
    frame = pd.DataFrame({
        "home_points_l3": rng.normal(28, 4, 30),
        "away_points_l3": rng.normal(25, 4, 30),
        "home_points_allowed_l3": rng.normal(22, 3, 30),
        "away_points_allowed_l3": rng.normal(24, 3, 30),
        "home_field_indicator": 1,
    })
    frame["home_target_points"] = 14 + frame["home_points_l3"] - 0.2 * frame["away_points_allowed_l3"]
    frame["away_target_points"] = 13 + frame["away_points_l3"] - 0.2 * frame["home_points_allowed_l3"]
    models = fit_score_models(frame)
    predicted = predict_matchups(frame, models)
    assert {"home_expected_points", "away_expected_points", "projected_total", "home_win_probability"}.issubset(predicted.columns)
    assert predicted["home_win_probability"].between(0, 1).all()


def test_model_feature_columns_are_unique_when_indicator_exists():
    frame = pd.DataFrame({
        "home_points_l3": [20.0],
        "away_points_l3": [18.0],
        "home_field_indicator": [1],
        "home_target_points": [21.0],
        "away_target_points": [17.0],
    })
    features = model_feature_columns(frame)
    assert features.count("home_field_indicator") == 1
    assert len(features) == len(set(features))


def test_normal_cdf_returns_expected_probabilities():
    probabilities = _normal_cdf(np.array([-1.0, 0.0, 1.0]))
    assert np.isclose(probabilities[1], 0.5)
    assert probabilities[0] < 0.5 < probabilities[2]
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
