"""Verified dataset assembly and training from an explicit dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .config import Paths
from .evaluation import evaluate_predictions
from .exports import export_all
from .features import build_matchup_features, build_team_pregame_features
from .ingestion.cfbd_canonical import PLAY_COUNT_BASIS
from .model import fit_score_models, predict_matchups
from .schemas import GAME_COLUMNS, TEAM_GAME_STAT_COLUMNS
from .validation import DataValidationError, validate_games, validate_team_game_stats

SNAPSHOT_SCHEMA = "canonical-basic-v1"
DATASET_SCHEMA = "dataset-basic-v1"
ARTIFACTS = ("games.csv", "team_game_stats.csv")
RED_ZONE_POLICY = "missing; excluded from baseline rolling features"


class DatasetError(RuntimeError):
    """Dataset verification, assembly, or training failed safely."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> str:
    content = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(content).hexdigest()


def _load_verified(directory: Path | str, schema: str, label: str):
    """Return (manifest, games, stats) only if hashes, rows, and schema verify."""
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise DatasetError(f"{label}: manifest is missing or unreadable") from None
    if not isinstance(manifest, dict) or manifest.get("status") != "ready" or manifest.get("schema_version") != schema:
        raise DatasetError(f"{label}: manifest must be status ready with schema {schema}")
    outputs = manifest.get("outputs")
    if (
        not isinstance(outputs, list)
        or any(not isinstance(entry, dict) for entry in outputs)
        or sorted(str(entry.get("file")) for entry in outputs) != sorted(ARTIFACTS)
    ):
        raise DatasetError(f"{label}: manifest outputs must be exactly {', '.join(ARTIFACTS)}")
    frames = {}
    for entry in outputs:
        path = directory / entry["file"]
        try:
            digest = _sha256(path)
        except OSError:
            raise DatasetError(f"{label}: missing artifact {entry['file']}") from None
        if digest != entry.get("sha256"):
            raise DatasetError(f"{label}: hash mismatch for {entry['file']}")
        try:
            frame = pd.read_csv(path)
        except ValueError:
            raise DatasetError(f"{label}: unreadable CSV {entry['file']}") from None
        if len(frame) != entry.get("rows"):
            raise DatasetError(f"{label}: row count mismatch for {entry['file']}")
        frames[entry["file"]] = frame
    games, stats = frames["games.csv"], frames["team_game_stats.csv"]
    if set(GAME_COLUMNS) - set(games.columns) or set(TEAM_GAME_STAT_COLUMNS) - set(stats.columns):
        raise DatasetError(f"{label}: required canonical columns are missing")
    return manifest, games, stats


def _clean(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _index(frame: pd.DataFrame, key: tuple[str, ...], label: str) -> dict:
    frame = frame.copy()
    if "game_date" in frame:
        frame["game_date"] = pd.to_datetime(frame["game_date"], errors="raise", utc=True)
    records = {}
    for row in frame.to_dict("records"):
        record = {name: _clean(value) for name, value in row.items()}
        identifier = tuple(record[name] for name in key)
        if identifier in records:
            raise DatasetError(f"{label}: duplicate record {identifier}")
        records[identifier] = record
    return records


def _check_consistency(games: dict, stats: dict) -> None:
    completed = {key[0]: record for key, record in games.items() if record["completed"]}
    if not completed:
        raise DatasetError("assembled dataset has no completed games")
    by_game: dict = {}
    for (game_id, team_id), record in stats.items():
        if game_id not in completed:
            raise DatasetError(f"team stats reference a non-completed game: {game_id}")
        by_game.setdefault(game_id, {})[team_id] = record
    for game_id, game in completed.items():
        teams = by_game.get(game_id, {})
        home, away = game["home_team_id"], game["away_team_id"]
        if set(teams) != {home, away}:
            raise DatasetError(f"completed game {game_id} lacks exactly two matching team records")
        for team_id, opponent, is_home, own, other in (
            (home, away, True, "home_score", "away_score"),
            (away, home, False, "away_score", "home_score"),
        ):
            record = teams[team_id]
            if (
                record["opponent_id"] != opponent or record["is_home"] != is_home
                or record["points"] != game[own] or record["points_allowed"] != game[other]
            ):
                raise DatasetError(f"team statistics disagree with game {game_id}")
    bases = {record.get("offensive_plays_basis") for record in stats.values()}
    if bases != {PLAY_COUNT_BASIS}:
        raise DatasetError("mixed or missing offensive-play basis")


def discover_snapshots(data_dir: Path | str) -> list[Path]:
    base = Path(data_dir) / "processed" / "cfbd"
    if not base.is_dir():
        return []
    return sorted(
        path for path in base.iterdir()
        if path.is_dir() and not path.name.startswith(".") and (path / "manifest.json").is_file()
    )


def assemble_dataset(snapshot_dirs, data_dir: Path | str) -> Path:
    """Verify snapshots, merge them conservatively, and publish a new dataset."""
    paths = [Path(item).resolve() for item in snapshot_dirs]
    if not paths:
        raise ValueError("at least one snapshot is required")
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate snapshot paths")
    loaded = []
    try:
        for path in paths:
            manifest, games, stats = _load_verified(path, SNAPSHOT_SCHEMA, f"snapshot {path.name}")
            if manifest.get("offensive_plays_basis") != PLAY_COUNT_BASIS:
                raise DatasetError(f"snapshot {path.name}: unsupported offensive-play basis")
            if not isinstance(manifest.get("started_at_utc"), str) or not isinstance(manifest.get("run_id"), str):
                raise DatasetError(f"snapshot {path.name}: manifest lacks run identity")
            loaded.append((
                manifest, path, _index(games, ("game_id",), path.name),
                _index(stats, ("game_id", "team_id"), path.name),
            ))
    except (ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, DatasetError):
            raise
        raise DatasetError("snapshot contains unreadable or inconsistent values") from None
    if len({item[0]["run_id"] for item in loaded}) != len(loaded):
        raise DatasetError("snapshots share a run ID")
    loaded.sort(key=lambda item: (item[0]["started_at_utc"], item[0]["run_id"]))

    merged_games, merged_stats, upgraded = {}, {}, 0
    for manifest, path, game_records, stat_records in loaded:
        for key, record in game_records.items():
            existing = merged_games.get(key)
            if existing is None or existing == record:
                merged_games.setdefault(key, record)
            elif not existing["completed"] and record["completed"]:
                merged_games[key] = record
                upgraded += 1
            elif existing["completed"] and not record["completed"]:
                continue
            else:
                raise DatasetError(f"conflicting game records for game_id {key[0]}")
        for key, record in stat_records.items():
            existing = merged_stats.get(key)
            if existing is None:
                merged_stats[key] = record
            elif existing != record:
                raise DatasetError(f"conflicting team statistics for {key}")
    _check_consistency(merged_games, merged_stats)

    extra = sorted({name for record in merged_games.values() for name in record} - set(GAME_COLUMNS))
    games_frame = pd.DataFrame(
        sorted(merged_games.values(), key=lambda row: (row["game_date"], row["game_id"])),
        columns=[*GAME_COLUMNS, *extra],
    )
    stats_frame = pd.DataFrame(
        sorted(merged_stats.values(), key=lambda row: (row["game_id"], row["team_id"])),
        columns=[*TEAM_GAME_STAT_COLUMNS, "offensive_plays_basis"],
    )
    games_checked = validate_games(games_frame)
    validate_team_game_stats(stats_frame, games_checked)

    dataset_id = _run_id()
    parent = Path(data_dir) / "datasets"
    staging, output = parent / ("." + dataset_id + ".staging"), parent / dataset_id
    created = False
    try:
        parent.mkdir(parents=True, exist_ok=True)
        staging.mkdir(exist_ok=False)
        created = True
        artifacts = []
        for filename, frame in (("games.csv", games_frame), ("team_game_stats.csv", stats_frame)):
            path = staging / filename
            frame.to_csv(path, index=False, na_rep="")
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            artifacts.append({"file": filename, "rows": len(frame), "sha256": _sha256(path)})
        completed = int(games_frame["completed"].astype(bool).sum())
        manifest = {
            "dataset_id": dataset_id, "schema_version": DATASET_SCHEMA, "status": "ready",
            "created_at_utc": _utc(),
            "inputs": [
                {
                    "run_id": item[0]["run_id"], "path": str(item[1]),
                    "manifest_sha256": _sha256(item[1] / "manifest.json"),
                    "started_at_utc": item[0]["started_at_utc"], "scope": item[0].get("scope"),
                    "counts": item[0].get("counts"),
                }
                for item in loaded
            ],
            "counts": {
                "snapshots": len(loaded), "games": len(games_frame), "completed_games": completed,
                "upcoming_games": len(games_frame) - completed, "team_stat_rows": len(stats_frame),
                "upcoming_replaced_by_completed": upgraded,
            },
            "outputs": artifacts, "offensive_plays_basis": PLAY_COUNT_BASIS,
            "red_zone_policy": RED_ZONE_POLICY,
        }
        _write_json(staging / "manifest.json", manifest)
        if output.exists():
            raise FileExistsError("dataset already exists")
        staging.rename(output)
        return output
    except Exception as exc:
        if created and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise DatasetError(f"Dataset assembly failed ({type(exc).__name__})") from None


def train_from_dataset(dataset_dir: Path | str, data_dir: Path | str) -> Path:
    """Train from one verified dataset; write only to a new run directory."""
    manifest, games, stats = _load_verified(dataset_dir, DATASET_SCHEMA, "dataset")
    games = validate_games(games)
    stats = validate_team_game_stats(stats, games)
    completed = games[games["completed"].astype(bool)]
    if completed[["season", "week"]].drop_duplicates().shape[0] < 2:
        raise DatasetError("training requires completed games from at least two season/week groups")
    run_dir = Path(data_dir) / "runs" / manifest["dataset_id"] / _run_id()
    run_dir.mkdir(parents=True, exist_ok=False)
    paths = Paths(
        root=run_dir, data_dir=Path(data_dir), export_dir=run_dir / "exports", model_dir=run_dir / "models",
    )
    team_features = build_team_pregame_features(games, stats)
    matchups = build_matchup_features(games, team_features)
    training = matchups[matchups["completed"].astype(bool)].copy()
    models = fit_score_models(training, paths)
    predictions = predict_matchups(matchups, models)
    performance = evaluate_predictions(predictions)
    historical = predictions[predictions["completed"].astype(bool)].copy()
    export_all(predictions, team_features, historical, performance, paths)
    _write_json(run_dir / "run_manifest.json", {
        "dataset_id": manifest["dataset_id"], "created_at_utc": _utc(),
        "dataset_manifest_sha256": _sha256(Path(dataset_dir) / "manifest.json"),
        "completed_games": len(completed), "prediction_rows": len(predictions),
        "evaluation_scope": "in_sample_pipeline_check",
        "note": "Metrics compare predictions with games used for training; they are not out-of-sample accuracy.",
        "exports_dir": str(paths.export_dir), "model_file": str(paths.model_file),
    })
    return run_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Assemble verified datasets and train from an explicit dataset")
    parser.add_argument("--data-dir", type=Path, default=Paths().data_dir)
    commands = parser.add_subparsers(dest="command", required=True)
    assemble = commands.add_parser("assemble", help="verify snapshots and publish a new dataset")
    group = assemble.add_mutually_exclusive_group(required=True)
    group.add_argument("--snapshot", action="append", type=Path)
    group.add_argument("--all", action="store_true", help="use every published CFBD snapshot")
    train = commands.add_parser("train", help="train from one assembled dataset")
    train.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "assemble":
            snapshots = discover_snapshots(args.data_dir) if args.all else args.snapshot
            output = assemble_dataset(snapshots, args.data_dir)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            print(f"Published dataset: {output}")
            print(f"Counts: {manifest['counts']}")
            print("Original snapshots and active model inputs were not changed.")
        else:
            run_dir = train_from_dataset(args.dataset, args.data_dir)
            print(f"Training run: {run_dir}")
            print(f"Workbook: {run_dir / 'exports' / 'cfb_forecast_export.xlsx'}")
            print("Metrics are in-sample pipeline checks, not out-of-sample accuracy.")
    except (DatasetError, DataValidationError, ValueError, OSError) as exc:
        print(f"Dataset workflow failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
