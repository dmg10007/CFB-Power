"""Pregame forecasts for upcoming games using the rolling-average model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .config import Paths
from .dataset_workflow import (
    DATASET_SCHEMA,
    DatasetError,
    _load_verified,
    _run_id,
    _sha256,
    _utc,
    _write_json,
)
from .model import _normal_cdf
from .power_ranking import fit_power_rankings
from .validation import DataValidationError, validate_games, validate_team_game_stats

MODEL_NAME = "power_rankings_v1"
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
            expected = next(
                entry["sha256"] for entry in snapshot["requests"] if entry["file"] == "games.json"
            )
            if _sha256(games_path) != expected:
                raise ValueError("raw schedule changed")
            for game in json.loads(games_path.read_text(encoding="utf-8")):
                for side in ("home", "away"):
                    team_id, name = game.get(f"{side}Id"), game.get(f"{side}Team")
                    if type(team_id) is int and isinstance(name, str) and name:
                        names[team_id] = name
        except (OSError, ValueError, KeyError, TypeError, StopIteration):
            warnings.append(
                f"team names unavailable from snapshot {item.get('run_id', '?')}; showing IDs where missing"
            )
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
    for column in (
        "home_expected_points",
        "away_expected_points",
        "projected_spread_home",
        "projected_total",
        "market_spread",
        "market_total",
        "spread_edge",
        "total_edge",
    ):
        if column in display.columns and pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].round(1)
    display["home_win_probability"] = display["home_win_probability"].round(3)
    sheet.append(list(display.columns))
    for row in display.itertuples(index=False, name=None):
        sheet.append(
            [None if isinstance(value, float) and np.isnan(value) else value for value in row]
        )
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
    dataset_dir: Path | str,
    data_dir: Path | str,
    as_of=None,
    min_history_games: int = DEFAULT_MIN_HISTORY_GAMES,
    model: str = "power_rankings",
    lines_data: list[dict[str, Any]] | None = None,
) -> Path:
    """Forecast not-yet-started games from completed games played before their kickoff.

    Supports:
    - 'power_rankings': Opponent-adjusted Ridge ratings with full situational efficiency metrics
      (yards/play, 3rd down, red zone, turnovers, TOP), SOS schedule adjustments, and recency weighting.
    - 'rolling_average': Baseline average of season points scored and opponent points allowed.
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
    league_home, league_away = (
        float(completed["home_score"].mean()),
        float(completed["away_score"].mean()),
    )
    history = stats.merge(completed[["game_id", "season", "game_date"]], on="game_id")
    by_team = {
        key: frame.sort_values("game_date") for key, frame in history.groupby(["team_id", "season"])
    }

    def state(team_id, season, kickoff):
        frame = by_team.get((team_id, season))
        prior = frame[frame["game_date"] < kickoff] if frame is not None else None
        if prior is None or prior.empty:
            return 0, np.nan, np.nan
        return len(prior), float(prior["points"].mean()), float(prior["points_allowed"].mean())

    warnings: list = []
    names = _team_names(manifest, warnings)

    is_power_model = str(model).lower().startswith("power")
    model_label = "power_rankings_v1" if is_power_model else "rolling_average"

    rankings_engine = None
    if is_power_model:
        try:
            rankings_engine = fit_power_rankings(completed, stats, team_names=names)
        except Exception as exc:
            warnings.append(f"Power rankings fit fallback to rolling average due to: {exc}")
            is_power_model = False
            model_label = "rolling_average"

    market_lines: dict[str | int | tuple[str, str], dict[str, Any]] = {}
    if lines_data is not None:
        try:
            from .market import parse_consensus_lines

            market_lines = parse_consensus_lines(lines_data)
        except Exception:
            market_lines = {}
    else:
        for item in manifest.get("inputs", []):
            try:
                snap_manifest = Path(item["path"]) / "manifest.json"
                if snap_manifest.is_file():
                    sm = json.loads(snap_manifest.read_text(encoding="utf-8"))
                    raw_arch = Path(sm.get("raw_archive", ""))
                    for lines_candidate in raw_arch.glob("lines*.json"):
                        if lines_candidate.is_file():
                            raw_lines = json.loads(lines_candidate.read_text(encoding="utf-8"))
                            from .market import parse_consensus_lines

                            market_lines.update(parse_consensus_lines(raw_lines))
            except Exception:
                pass

    rows, skipped = [], []
    for game in upcoming.sort_values(["game_date", "game_id"]).itertuples(index=False):
        if not (pd.isna(game.home_score) and pd.isna(game.away_score)):
            skipped.append({"game_id": game.game_id, "reason": "has scores but is not completed"})
        elif game.game_date <= moment:
            skipped.append(
                {"game_id": game.game_id, "reason": "kickoff has passed or game is in progress"}
            )
        else:
            home_n, home_offense, home_defense = state(
                game.home_team_id, game.season, game.game_date
            )
            away_n, away_offense, away_defense = state(
                game.away_team_id, game.season, game.game_date
            )

            home_team_name = names.get(game.home_team_id)
            away_team_name = names.get(game.away_team_id)
            is_neutral = bool(game.neutral_site)
            matchup_str = (
                f"{away_team_name or game.away_team_id} {'vs' if is_neutral else '@'} "
                f"{home_team_name or game.home_team_id}{' (N)' if is_neutral else ''}"
            )

            if is_power_model and rankings_engine is not None and (home_n > 0 or away_n > 0):
                comp = rankings_engine.compare_teams(
                    game.home_team_id, game.away_team_id, neutral_site=is_neutral
                )
                home = comp.team_a_expected_points
                away = comp.team_b_expected_points
                spread = comp.projected_spread
                total = comp.projected_total
                prob = comp.team_a_win_probability
                score_low_h = comp.team_a_score_low
                score_high_h = comp.team_a_score_high
                score_low_a = comp.team_b_score_low
                score_high_a = comp.team_b_score_high
                conf_tier = comp.confidence_tier
            else:
                home = _mean_or([home_offense, away_defense], league_home)
                away = _mean_or([away_offense, home_defense], league_away)
                spread = home - away
                total = home + away
                prob = float(_normal_cdf((home - away) / scale))
                score_low_h = max(0.0, round(home - scale / 1.4, 1))
                score_high_h = round(home + scale / 1.4, 1)
                score_low_a = max(0.0, round(away - scale / 1.4, 1))
                score_high_a = round(away + scale / 1.4, 1)
                conf_tier = "Low"

            from .market import compute_market_edges

            line_entry = (
                market_lines.get(game.game_id)
                or market_lines.get(str(game.game_id))
                or (
                    home_team_name
                    and away_team_name
                    and market_lines.get((home_team_name.lower(), away_team_name.lower()))
                )
                or {}
            )
            m_spread = line_entry.get("market_spread")
            m_total = line_entry.get("market_total")
            edges = compute_market_edges(
                projected_home_points=home,
                projected_away_points=away,
                market_spread=m_spread,
                market_total=m_total,
            )

            rows.append(
                {
                    "game_id": game.game_id,
                    "kickoff_utc": game.game_date.strftime("%Y-%m-%d %H:%M UTC"),
                    "matchup": matchup_str,
                    "home_team_id": game.home_team_id,
                    "away_team_id": game.away_team_id,
                    "home_team": home_team_name,
                    "away_team": away_team_name,
                    "neutral_site": is_neutral,
                    "home_games_played": home_n,
                    "away_games_played": away_n,
                    "low_history": min(home_n, away_n) < min_history_games,
                    "home_expected_points": home,
                    "away_expected_points": away,
                    "projected_spread_home": spread,
                    "projected_total": total,
                    "home_win_probability": prob,
                    "home_score_low": score_low_h,
                    "home_score_high": score_high_h,
                    "away_score_low": score_low_a,
                    "away_score_high": score_high_a,
                    "confidence_tier": conf_tier,
                    "market_spread": m_spread,
                    "market_total": m_total,
                    "spread_edge": edges["spread_edge"],
                    "total_edge": edges["total_edge"],
                    "value_tier": edges["value_tier"],
                    "model": model_label,
                }
            )
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
    unresolved = sum(1 for item in skipped if item["reason"].startswith("kickoff has passed"))
    if unresolved:
        warnings.append(
            f"{unresolved} games are past kickoff with no recorded result (in progress, or results not yet captured); "
            "team histories exclude them, so recapture that week before relying on these forecasts"
        )
    low = int(predictions["low_history"].sum())
    if low:
        warnings.append(
            f"{low} of {len(predictions)} games involve a team with fewer than {min_history_games} prior games"
        )

    run_dir = Path(data_dir) / "runs" / manifest["dataset_id"] / ("predictions_" + _run_id())
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(skipped, columns=["game_id", "reason"]).to_csv(
        run_dir / "skipped.csv", index=False
    )
    model_desc = (
        "power_rankings_v1: Opponent-adjusted Ridge ratings with situational efficiency (yards/play, 3rd down, red zone, turnovers, TOP), SOS schedule adjustment, and recency weighting"
        if is_power_model
        else "rolling_average: average of a team's season points scored and the opponent's season points allowed"
    )
    hfa_desc = (
        "+2.5 points explicit home-field advantage (neutral-site games adjusted to 0.0)"
        if is_power_model
        else "No home-field adjustment is applied; 'home' is the listed home team."
    )
    readme = [
        ["CFB Power upcoming-game forecasts"],
        ["Generated UTC", _utc()],
        ["As-of time UTC", moment.isoformat()],
        ["Data through (UTC)", data_through.isoformat()],
        ["Model", model_desc],
        ["Home field", hfa_desc],
        [
            "Win probability",
            "Normal distribution CDF based on projected scoring margin and model residual error standard deviation.",
        ],
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
    _write_json(
        run_dir / "manifest.json",
        {
            "kind": "upcoming_predictions",
            "status": "ready",
            "created_at_utc": _utc(),
            "dataset_id": manifest["dataset_id"],
            "dataset_manifest_sha256": _sha256(Path(dataset_dir) / "manifest.json"),
            "model": model_label,
            "as_of_utc": moment.isoformat(),
            "data_through_utc": data_through.isoformat(),
            "settings": {"min_history_games": min_history_games, "stale_days": STALE_DAYS},
            "counts": {
                "upcoming_in_dataset": int(len(upcoming)),
                "predicted": int(len(predictions)),
                "skipped": len(skipped),
                "low_history": low,
                "past_kickoff_unresolved": unresolved,
            },
            "skip_reasons": reasons,
            "warnings": warnings,
            "outputs": outputs,
            "notes": [
                "Power rankings model incorporates opponent adjustments, SOS, situational box stats, and recency weighting."
                if is_power_model
                else "Rolling-average model selected from walk-forward evaluation; see docs/UPCOMING_PREDICTIONS.md."
            ],
        },
    )
    return run_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Forecast upcoming games from one verified dataset"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Paths().data_dir)
    parser.add_argument("--as-of", help="ISO timestamp with timezone; defaults to now (UTC)")
    parser.add_argument("--min-history-games", type=int, default=DEFAULT_MIN_HISTORY_GAMES)
    parser.add_argument(
        "--model",
        default="power_rankings",
        choices=["power_rankings", "rolling_average"],
        help="Forecasting model to use (default: power_rankings)",
    )
    args = parser.parse_args(argv)
    try:
        run_dir = predict_upcoming(
            args.dataset,
            args.data_dir,
            args.as_of,
            args.min_history_games,
            model=args.model,
        )
    except (DatasetError, DataValidationError, ValueError, OSError) as exc:
        print(f"Upcoming predictions failed: {exc}", file=sys.stderr)
        return 1
    predictions = pd.read_csv(run_dir / "predictions.csv").fillna(
        {"home_team": "", "away_team": ""}
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    print(f"Predictions run: {run_dir}")
    print(f"Workbook: {run_dir / 'upcoming_predictions.xlsx'}")
    print(
        f"Predicted {manifest['counts']['predicted']} games; skipped {manifest['counts']['skipped']}"
    )
    table = pd.DataFrame(
        {
            "kickoff": predictions["kickoff_utc"],
            "matchup": [
                f"{away or away_id} @ {home or home_id}"
                for away, home, away_id, home_id in zip(
                    predictions["away_team"],
                    predictions["home_team"],
                    predictions["away_team_id"],
                    predictions["home_team_id"],
                    strict=True,
                )
            ],
            "spread_home": predictions["projected_spread_home"].round(1),
            "total": predictions["projected_total"].round(1),
            "home_win_pct": (predictions["home_win_probability"] * 100).round(1),
            "low_history": predictions["low_history"],
        }
    )
    print(table.head(25).to_string(index=False))
    for warning in manifest["warnings"]:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
