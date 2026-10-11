"""Chronological walk-forward evaluation from one verified dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Paths
from .dataset_workflow import DATASET_SCHEMA, DatasetError, _load_verified, _run_id, _sha256, _utc, _write_json
from .evaluation import calibration_bins
from .features import build_matchup_features, build_team_pregame_features
from .model import _normal_cdf, fit_score_models, predict_matchups
from .model_candidates import CANDIDATE_DEFINITIONS, CANDIDATE_NAMES, CANDIDATES, add_prior_game_counts
from .validation import DataValidationError, validate_games, validate_team_game_stats

MODEL_NAMES = ("ridge", "train_mean", "rolling_average")
MIN_TRAINING_GAMES = 20
SMALL_SAMPLE_GAMES = 100
IMPROVEMENT_METRICS = ("spread_mae", "total_mae", "brier_score")


def _metrics(frame: pd.DataFrame, name: str) -> dict:
    home, away = frame[f"{name}_home_points"], frame[f"{name}_away_points"]
    home_error, away_error = home - frame["home_score"], away - frame["away_score"]
    predicted_margin = home - away
    actual_margin = frame["home_score"] - frame["away_score"]
    probability = frame[f"{name}_home_win_probability"].clip(0.001, 0.999)
    return {
        "model": name,
        "games": int(len(frame)),
        "score_mae": float((home_error.abs().mean() + away_error.abs().mean()) / 2),
        "score_rmse": float(np.sqrt((np.square(home_error).mean() + np.square(away_error).mean()) / 2)),
        "spread_mae": float((predicted_margin - actual_margin).abs().mean()),
        "total_mae": float(((home + away) - (frame["home_score"] + frame["away_score"])).abs().mean()),
        "brier_score": float(np.mean(np.square(probability - (actual_margin > 0).astype(float)))),
        "winner_accuracy": float(((predicted_margin > 0) == (actual_margin > 0)).mean()),
    }


def _improvement(overall: pd.DataFrame, metric: str, model: str, baseline: str) -> float | None:
    values = overall.set_index("model")[metric]
    if values[baseline] == 0:
        return None
    return float((values[baseline] - values[model]) / values[baseline] * 100)


def walk_forward_evaluate(
    dataset_dir: Path | str, data_dir: Path | str, min_train_periods: int = 2, candidates: bool = False,
) -> Path:
    """Train on earlier periods, predict the next, and score out-of-sample.

    Features come from one pass of the existing builder, which shifts every rolling
    aggregate so a game uses only its teams' earlier games. Each period's model is fit
    only on completed games from earlier periods. Writes only to a new run directory.
    With candidates=True, extra experimental models are scored on the same games.
    """
    if type(min_train_periods) is not int or min_train_periods < 1:
        raise ValueError("min_train_periods must be a positive integer")
    names = MODEL_NAMES + (CANDIDATE_NAMES if candidates else ())
    manifest, games, stats = _load_verified(dataset_dir, DATASET_SCHEMA, "dataset")
    games = validate_games(games)
    stats = validate_team_game_stats(stats, games)
    team_features = build_team_pregame_features(games, stats)
    matchups = add_prior_game_counts(build_matchup_features(games, team_features), team_features)
    completed = matchups[matchups["completed"].astype(bool)].copy()
    kind = completed["season_type"].astype(str) if "season_type" in completed.columns else "regular"
    completed["period"] = completed["season"].astype(str) + " " + kind + " week " + completed["week"].astype(str)
    order = completed.groupby("period")["game_date"].min().sort_values(kind="stable")
    periods = list(order.index)

    parts, tested, skipped = [], [], []
    for index, period in enumerate(periods):
        if index < min_train_periods:
            skipped.append({"period": period, "reason": "fewer than the minimum earlier periods"})
            continue
        train = completed[completed["period"].isin(periods[:index])]
        target = completed[completed["period"] == period]
        if len(train) < MIN_TRAINING_GAMES:
            skipped.append({"period": period, "reason": f"fewer than {MIN_TRAINING_GAMES} training games"})
            continue
        models = fit_score_models(train)
        predicted = predict_matchups(target, models)
        margins = train["home_score"] - train["away_score"]
        scale = max(float(margins.std(ddof=1)), 1.0)
        train_home, train_away = float(train["home_score"].mean()), float(train["away_score"].mean())
        home_rate = float(np.clip((margins > 0).mean(), 0.001, 0.999))
        rolling_home = target[["home_points_season", "away_points_allowed_season"]].mean(axis=1).fillna(train_home)
        rolling_away = target[["away_points_season", "home_points_allowed_season"]].mean(axis=1).fillna(train_away)
        columns = {
            "game_id": target["game_id"], "period": period, "game_date": target["game_date"],
            "home_team_id": target["home_team_id"], "away_team_id": target["away_team_id"],
            "home_score": target["home_score"], "away_score": target["away_score"],
            "train_games": len(train),
            "ridge_home_points": predicted["home_expected_points"],
            "ridge_away_points": predicted["away_expected_points"],
            "ridge_home_win_probability": predicted["home_win_probability"],
            "train_mean_home_points": train_home, "train_mean_away_points": train_away,
            "train_mean_home_win_probability": home_rate,
            "rolling_average_home_points": rolling_home, "rolling_average_away_points": rolling_away,
            "rolling_average_home_win_probability": _normal_cdf((rolling_home - rolling_away) / scale),
        }
        if candidates:
            for name, candidate in CANDIDATES.items():
                home, away, probability = candidate(train, target)
                columns[f"{name}_home_points"] = home
                columns[f"{name}_away_points"] = away
                columns[f"{name}_home_win_probability"] = probability
        parts.append(pd.DataFrame(columns))
        tested.append({
            "period": period, "train_games": int(len(train)), "test_games": int(len(target)),
            "train_through": periods[index - 1],
        })
    if not parts:
        raise DatasetError(
            f"no evaluable periods: need {min_train_periods} earlier periods and {MIN_TRAINING_GAMES} training games"
        )
    predictions = pd.concat(parts).sort_values(["game_date", "game_id"]).reset_index(drop=True)
    weekly = pd.DataFrame([
        {"period": period, **_metrics(group, name)}
        for period, group in predictions.groupby("period", sort=False) for name in names
    ])
    overall = pd.DataFrame([_metrics(predictions, name) for name in names])
    calibration = calibration_bins(pd.DataFrame({
        "game_id": predictions["game_id"], "home_score": predictions["home_score"],
        "away_score": predictions["away_score"],
        "home_win_probability": predictions["ridge_home_win_probability"],
    }))
    improvement = {
        f"{metric}_vs_{baseline}": _improvement(overall, metric, "ridge", baseline)
        for metric in IMPROVEMENT_METRICS for baseline in MODEL_NAMES[1:]
    }
    if candidates:
        for name in CANDIDATE_NAMES:
            for metric in IMPROVEMENT_METRICS:
                improvement[f"{metric}_{name}_vs_rolling_average"] = _improvement(
                    overall, metric, name, "rolling_average"
                )
    warnings = []
    if len(predictions) < SMALL_SAMPLE_GAMES:
        warnings.append(
            f"small sample: {len(predictions)} out-of-sample games over {len(tested)} periods; metrics are noisy"
        )
    notes = [
        "Positive improvement means the model had lower error than that baseline.",
        "Each game's features use only its teams' earlier games; each period's model is fit on earlier periods only.",
        "A team playing twice inside one period could use the first game as pregame information for the second.",
    ]
    if candidates:
        notes.append(
            "Several candidate models were scored on the same games. The best-looking candidate may reflect chance "
            "on a small sample; confirm on later weeks before relying on it."
        )

    run_dir = Path(data_dir) / "runs" / manifest["dataset_id"] / ("walk_forward_" + _run_id())
    run_dir.mkdir(parents=True, exist_ok=False)
    outputs = []
    for filename, frame in (
        ("predictions.csv", predictions), ("weekly_metrics.csv", weekly),
        ("overall_metrics.csv", overall), ("calibration.csv", calibration),
    ):
        frame.to_csv(run_dir / filename, index=False)
        outputs.append({"file": filename, "rows": int(len(frame)), "sha256": _sha256(run_dir / filename)})
    _write_json(run_dir / "manifest.json", {
        "kind": "walk_forward", "status": "ready", "created_at_utc": _utc(),
        "dataset_id": manifest["dataset_id"],
        "dataset_manifest_sha256": _sha256(Path(dataset_dir) / "manifest.json"),
        "evaluation_scope": "out_of_sample_walk_forward",
        "settings": {"min_train_periods": min_train_periods, "min_training_games": MIN_TRAINING_GAMES},
        "candidates": {"enabled": candidates, "definitions": CANDIDATE_DEFINITIONS if candidates else {}},
        "counts": {
            "periods": len(periods), "tested_periods": len(tested), "skipped_periods": len(skipped),
            "tested_games": int(len(predictions)),
        },
        "tested": tested, "skipped": skipped,
        "overall": overall.to_dict("records"),
        "ridge_improvement_vs_baselines_pct": improvement,
        "warnings": warnings, "outputs": outputs, "notes": notes,
    })
    return run_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Walk-forward evaluation from one verified dataset")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Paths().data_dir)
    parser.add_argument("--min-train-periods", type=int, default=2)
    parser.add_argument("--with-candidates", action="store_true", help="also score experimental candidate models")
    args = parser.parse_args(argv)
    try:
        run_dir = walk_forward_evaluate(args.dataset, args.data_dir, args.min_train_periods, args.with_candidates)
    except (DatasetError, DataValidationError, ValueError, OSError) as exc:
        print(f"Walk-forward evaluation failed: {exc}", file=sys.stderr)
        return 1
    overall = pd.read_csv(run_dir / "overall_metrics.csv")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    print(f"Walk-forward run: {run_dir}")
    print(f"Out-of-sample games: {manifest['counts']['tested_games']} over {manifest['counts']['tested_periods']} periods")
    print(overall.round(3).to_string(index=False))
    for warning in manifest["warnings"]:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
