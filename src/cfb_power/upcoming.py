"""Pregame forecasts for upcoming games using the rolling-average model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .config import Paths
from .dataset_workflow import DATASET_SCHEMA, DatasetError, _load_verified, _run_id, _sha256, _utc, _write_json
from .model import _normal_cdf
from .validation import DataValidationError, validate_games, validate_team_game_stats

MODEL_NAME = "rolling_average"
STALE_DAYS = 9
DEFAULT_MIN_HISTORY_GAMES = 3


def _team_names(dataset_manifest: dict, warnings: list) -> dict:
    """Read team display names from hash-verified raw schedule archives, if still present."""
    names: dict = {}
    for item in dataset_manifest.get("inputs", []):
        try:
            snapshot_manifest = Path(item["path"]) / "manifest.json"
            if _sha256(snapshot_manifest) != item["manifest_sha256"]:
                raise ValueError("snapshot manifest changed")
            snapshot = json.loads(snapshot_manifest.read_text(encoding="utf-8"))
            games_path = Path(snapshot["raw_archive"]) / "games.json"
            expected = next(entry["sha256"] for entry in snapshot["requests"] if entry["file"] == "games.json")
            if _sha256(games_path) != expected:
                raise ValueError("raw schedule changed")
            for game in json.loads(games_path.read_text(encoding="utf-8")):
                for side in ("home", "away"):
                    team_id, name = game.get(f"{side}Id"), game.get(f"{side}Team")
                    if type(team_id) is int and isinstance(name, str) and name:
                        names[team_id] = name
        except (OSError, ValueError, KeyError, TypeError, StopIteration):
            warnings.append(f"team names unavailable from snapshot {item.get('run_id', '?')}; showing IDs where missing")
    return names


def _as_of(value) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz="UTC")
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError):
        raise ValueError("as_of must be an ISO timestamp with a timezone") from None
    if stamp.tzinfo is None:
        raise ValueError("as_of must include a timezone, for example 2026-10-10T18:00:00Z")
    return stamp.tz_convert("UTC")


def _mean_or(values, fallback: float) -> float:
    present = [value for value in values if not np.isnan(value)]
    return float(np.mean(present)) if present else fallback


def _format_workbook(path: Path, predictions: pd.DataFrame, readme: list) -> None:
    workbook = Workbook()
    workbook.remove(workbook.active)
    sheet = workbook.create_sheet("README")
    for row in readme:
        sheet.append(row)
    sheet["A1"].font = Font(size=16, bold=True, color="FFFFFF")
    sheet["A1"].fill = PatternFill("solid", fgColor="123047")
    sheet.column_dimensions["A"].width = 24
    sheet.column_dimensions["B"].width = 110
    sheet = workbook.create_sheet("Predictions")
    display = predictions.copy()
    for column in ("home_expected_points", "away_expected_points", "projected_spread_home", "projected_total"):
        display[column] = display[column].round(1)
    display["home_win_probability"] = display["home_win_probability"].round(3)
    sheet.append(list(display.columns))
    for row in display.itertuples(index=False, name=None):
        sheet.append([None if isinstance(value, float) and np.isnan(value) else value for value in row])
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for cell in sheet[1]:
        cell.font = Font(color="FFFFFF", bold=True)
        cell.fill = PatternFill("solid", fgColor="123047")
        cell.alignment = Alignment(horizontal="center")
    for index, column in enumerate(display.columns, start=1):
        width = max(len(str(column)), *(len(str(value)) for value in display[column].head(100))) + 2
        sheet.column_dimensions[get_column_letter(index)].width = min(width, 30)
        if column == "home_win_probability":
            for cell in sheet[get_column_letter(index)][1:]:
                cell.number_format = "0.0%"
    workbook.save(path)


def predict_upcoming(
    dataset_dir: Path | str, data_dir: Path | str, as_of=None,
    min_history_games: int = DEFAULT_MIN_HISTORY_GAMES,
) -> Path:
    """Forecast not-yet-started games from completed games played before their kickoff.

    Uses the rolling-average model: each side's expected points are the average of its own
    season points scored and the opponent's season points allowed (league mean when a team
    has no history). There is no home-field term. Writes only to a new run directory.
    """
    if type(min_history_games) is not int or min_history_games < 0:
        raise ValueError("min_history_games must be a nonnegative integer")
    moment = _as_of(as_of)
    manifest, games, stats = _load_verified(dataset_dir, DATASET_SCHEMA, "dataset")
    games = validate_games(games)
    stats = validate_team_game_stats(stats, games)
    done = games["completed"].astype(bool)
    completed, upcoming = games[done], games[~done]
    if completed.empty:
        raise DatasetError("dataset has no completed games to learn from")

    margins = completed["home_score"] - completed["away_score"]
    scale = max(float(margins.std(ddof=1)), 1.0) if len(completed) > 1 else 1.0
    league_home, league_away = float(completed["home_score"].mean()), float(completed["away_score"].mean())
    history = stats.merge(completed[["game_id", "season", "game_date"]], on="game_id")
    by_team = {key: frame.sort_values("game_date") for key, frame in history.groupby(["team_id", "season"])}

    def state(team_id, season, kickoff):
        frame = by_team.get((team_id, season))
        prior = frame[frame["game_date"] < kickoff] if frame is not None else None
        if prior is None or prior.empty:
            return 0, np.nan, np.nan
        return len(prior), float(prior["points"].mean()), float(prior["points_allowed"].mean())

    warnings: list = []
    names = _team_names(manifest, warnings)
    rows, skipped = [], []
    for game in upcoming.sort_values(["game_date", "game_id"]).itertuples(index=False):
        if not (pd.isna(game.home_score) and pd.isna(game.away_score)):
            skipped.append({"game_id": game.game_id, "reason": "has scores but is not completed"})
        elif game.game_date <= moment:
            skipped.append({"game_id": game.game_id, "reason": "kickoff has passed or game is in progress"})
        else:
            home_n, home_offense, home_defense = state(game.home_team_id, game.season, game.game_date)
            away_n, away_offense, away_defense = state(game.away_team_id, game.season, game.game_date)
            home = _mean_or([home_offense, away_defense], league_home)
            away = _mean_or([away_offense, home_defense], league_away)
            rows.append({
                "game_id": game.game_id,
                "kickoff_utc": game.game_date.strftime("%Y-%m-%d %H:%M UTC"),
                "home_team_id": game.home_team_id, "away_team_id": game.away_team_id,
                "home_team": names.get(game.home_team_id, ""), "away_team": names.get(game.away_team_id, ""),
                "neutral_site": bool(game.neutral_site),
                "home_games_played": home_n, "away_games_played": away_n,
                "low_history": min(home_n, away_n) < min_history_games,
                "home_expected_points": home, "away_expected_points": away,
                "projected_spread_home": home - away, "projected_total": home + away,
                "home_win_probability": float(_normal_cdf((home - away) / scale)),
                "model": MODEL_NAME,
            })
    if not rows:
        reasons = sorted({item["reason"] for item in skipped}) or ["dataset has no upcoming games"]
        raise DatasetError("no upcoming games to predict (" + "; ".join(reasons) + ")")
    predictions = pd.DataFrame(rows)
    data_through = completed["game_date"].max()
    age_days = (moment - data_through).total_seconds() / 86400
    if age_days > STALE_DAYS:
        warnings.append(
            f"latest completed game is {age_days:.0f} days before as-of time; capture recent weeks before relying on these forecasts"
        )
    low = int(predictions["low_history"].sum())
    if low:
        warnings.append(f"{low} of {len(predictions)} games involve a team with fewer than {min_history_games} prior games")

    run_dir = Path(data_dir) / "runs" / manifest["dataset_id"] / ("predictions_" + _run_id())
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(skipped, columns=["game_id", "reason"]).to_csv(run_dir / "skipped.csv", index=False)
    readme = [
        ["CFB Power upcoming-game forecasts"],
        ["Generated UTC", _utc()],
        ["As-of time UTC", moment.isoformat()],
        ["Data through (UTC)", data_through.isoformat()],
        ["Model", "rolling_average: average of a team's season points scored and the opponent's season points allowed"],
        ["Home field", "No home-field adjustment is applied; 'home' is the listed home team."],
        ["Win probability", "Normal approximation using the spread of completed-game margins; treat as a rough guide."],
        ["Warnings", "; ".join(warnings) if warnings else "none"],
    ]
    _format_workbook(run_dir / "upcoming_predictions.xlsx", predictions, readme)
    outputs = [
        {"file": name, "sha256": _sha256(run_dir / name)}
        for name in ("predictions.csv", "skipped.csv", "upcoming_predictions.xlsx")
    ]
    reasons: dict = {}
    for item in skipped:
        reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
    _write_json(run_dir / "manifest.json", {
        "kind": "upcoming_predictions", "status": "ready", "created_at_utc": _utc(),
        "dataset_id": manifest["dataset_id"],
        "dataset_manifest_sha256": _sha256(Path(dataset_dir) / "manifest.json"),
        "model": MODEL_NAME, "as_of_utc": moment.isoformat(), "data_through_utc": data_through.isoformat(),
        "settings": {"min_history_games": min_history_games, "stale_days": STALE_DAYS},
        "counts": {
            "upcoming_in_dataset": int(len(upcoming)), "predicted": int(len(predictions)),
            "skipped": len(skipped), "low_history": low,
        },
        "skip_reasons": reasons, "warnings": warnings, "outputs": outputs,
        "notes": [
            "Rolling-average model selected from walk-forward evaluation; see docs/UPCOMING_PREDICTIONS.md.",
            "No home-field term: adding one worsened out-of-sample results in the model experiments.",
        ],
    })
    return run_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Forecast upcoming games from one verified dataset")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Paths().data_dir)
    parser.add_argument("--as-of", help="ISO timestamp with timezone; defaults to now (UTC)")
    parser.add_argument("--min-history-games", type=int, default=DEFAULT_MIN_HISTORY_GAMES)
    args = parser.parse_args(argv)
    try:
        run_dir = predict_upcoming(args.dataset, args.data_dir, args.as_of, args.min_history_games)
    except (DatasetError, DataValidationError, ValueError, OSError) as exc:
        print(f"Upcoming predictions failed: {exc}", file=sys.stderr)
        return 1
    predictions = pd.read_csv(run_dir / "predictions.csv").fillna({"home_team": "", "away_team": ""})
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    print(f"Predictions run: {run_dir}")
    print(f"Workbook: {run_dir / 'upcoming_predictions.xlsx'}")
    print(f"Predicted {manifest['counts']['predicted']} games; skipped {manifest['counts']['skipped']}")
    table = pd.DataFrame({
        "kickoff": predictions["kickoff_utc"],
        "matchup": [
            f"{away or away_id} @ {home or home_id}"
            for away, home, away_id, home_id in zip(
                predictions["away_team"], predictions["home_team"],
                predictions["away_team_id"], predictions["home_team_id"],
            )
        ],
        "spread_home": predictions["projected_spread_home"].round(1),
        "total": predictions["projected_total"].round(1),
        "home_win_pct": (predictions["home_win_probability"] * 100).round(1),
        "low_history": predictions["low_history"],
    })
    print(table.head(25).to_string(index=False))
    for warning in manifest["warnings"]:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
