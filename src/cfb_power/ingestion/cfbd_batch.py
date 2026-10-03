"""Audited CFBD batch capture and immutable canonical snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from cfb_power.config import Paths
from cfb_power.ingestion.cfbd_canonical import PLAY_COUNT_BASIS, normalize_cfbd_canonical, normalize_games
from cfb_power.ingestion.cfbd_client import CfbdClient
from cfb_power.validation import DataValidationError


class BatchIngestionError(RuntimeError):
    """A failed capture must not become a published dataset."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict | list) -> str:
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


def _select(payload, season, season_type, week):
    if not isinstance(payload, list) or any(not isinstance(game, dict) for game in payload):
        raise DataValidationError("schedule must be a list of objects")
    selected, exclusions, seen = [], [], set()
    for game in payload:
        game_id = game.get("id")
        if type(game_id) is not int or game_id <= 0 or game_id in seen:
            raise DataValidationError("invalid or duplicate schedule game ID")
        seen.add(game_id)
        if game.get("season") != season or game.get("seasonType") != season_type:
            raise DataValidationError("schedule response is outside requested season/type")
        if type(game.get("week")) is not int or game["week"] < 0:
            raise DataValidationError("invalid schedule week")
        if week is not None and game["week"] != week:
            raise DataValidationError("schedule response is outside requested week")
        if type(game.get("completed")) is not bool:
            raise DataValidationError("invalid completed flag")
        classes = [game.get("homeClassification"), game.get("awayClassification")]
        if any(value not in ("fbs", "fcs") for value in classes):
            raise DataValidationError("missing or unsupported team classification")
        reason = None
        if "fcs" in classes:
            reason = "non_fbs_matchup"
        elif not game["completed"]:
            if game.get("homePoints") is not None or game.get("awayPoints") is not None:
                reason = "unfinished_with_scores"
            elif game.get("startTimeTBD") is True or game.get("startDate") is None:
                reason = "unconfirmed_upcoming_kickoff"
        if reason:
            exclusions.append({"game_id": game_id, "reason": reason})
        else:
            selected.append(game)
    if not selected:
        raise DataValidationError("no eligible games in requested scope")
    return selected, exclusions


def ingest_cfbd_batch(
    *, season: int, season_type: str, week: int | None = None,
    data_dir: Path | str, max_completed_games: int = 100, client=None,
) -> Path:
    """Publish a new snapshot; never activate it or overwrite model input CSVs.

    Makes one schedule request plus one ID-only box request per completed game.
    Returns the published snapshot directory. Failed runs retain raw audit data.
    """
    if type(season) is not int or season < 1:
        raise ValueError("season must be a positive integer")
    if season_type not in ("regular", "postseason"):
        raise ValueError("season_type must be regular or postseason")
    if week is not None and (type(week) is not int or week < 0):
        raise ValueError("week must be a nonnegative integer")
    if type(max_completed_games) is not int or max_completed_games < 1:
        raise ValueError("max_completed_games must be positive")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex
    root = Path(data_dir)
    raw = root / "raw" / "cfbd" / str(season) / run_id
    parent = root / "processed" / "cfbd"
    staging, output = parent / ("." + run_id + ".staging"), parent / run_id
    raw.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": run_id, "source": "CFBD", "schema_version": "canonical-basic-v1",
        "started_at_utc": _utc(), "status": "capturing",
        "scope": {"season": season, "season_type": season_type, "week": week},
        "max_completed_games": max_completed_games, "requests": [], "exclusions": [],
        "offensive_plays_basis": PLAY_COUNT_BASIS,
        "red_zone_policy": "missing; excluded from baseline rolling features",
    }
    audit = raw / "manifest.json"
    created_staging = False
    try:
        _write_json(audit, manifest)
        api = client if client is not None else CfbdClient()

        def capture(endpoint, params, filename):
            payload = api.get_json(endpoint, params)
            digest = _write_json(raw / filename, payload)
            manifest["requests"].append({
                "endpoint": endpoint, "params": params, "file": filename,
                "retrieved_at_utc": _utc(), "sha256": digest,
            })
            _write_json(audit, manifest)
            return payload

        params = {"year": season, "seasonType": season_type, "classification": "fbs"}
        if week is not None:
            params["week"] = week
        schedule = capture("/games", params, "games.json")
        selected, exclusions = _select(schedule, season, season_type, week)
        manifest["exclusions"] = exclusions
        normalize_games(selected)
        completed = [game for game in selected if game["completed"]]
        manifest["counts"] = {
            "source_games": len(schedule), "selected_games": len(selected),
            "completed_games": len(completed), "upcoming_games": len(selected) - len(completed),
            "excluded_games": len(exclusions),
        }
        _write_json(audit, manifest)
        if len(completed) > max_completed_games:
            raise DataValidationError("completed-game request limit exceeded")
        boxes = []
        for game in completed:
            game_id = game["id"]
            response = capture("/games/teams", {"id": game_id}, f"box_{game_id}.json")
            if not isinstance(response, list) or len(response) != 1:
                raise DataValidationError("ID-only box request must return exactly one game")
            if not isinstance(response[0], dict) or response[0].get("id") != game_id:
                raise DataValidationError("box response does not match requested game")
            boxes.extend(response)
        games, stats = normalize_cfbd_canonical(selected, boxes)
        games["season_type"] = season_type
        manifest["counts"]["team_stat_rows"] = len(stats)
        parent.mkdir(parents=True, exist_ok=True)
        staging.mkdir(exist_ok=False)
        created_staging = True
        outputs = []
        for filename, frame in (("games.csv", games), ("team_game_stats.csv", stats)):
            path = staging / filename
            frame.to_csv(path, index=False, na_rep="")
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            outputs.append({"file": filename, "rows": len(frame), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        manifest.update(status="validated", validated_at_utc=_utc(), outputs=outputs)
        _write_json(audit, manifest)
        published_manifest = {**manifest, "status": "ready", "raw_archive": str(raw.resolve())}
        _write_json(staging / "manifest.json", published_manifest)
        if output.exists():
            raise FileExistsError("snapshot already exists")
        staging.rename(output)
        return output
    except Exception as exc:
        if created_staging and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        manifest.update(status="failed", failed_at_utc=_utc(), error_type=type(exc).__name__)
        try:
            _write_json(audit, manifest)
        except OSError:
            pass
        raise BatchIngestionError(f"Ingestion failed ({type(exc).__name__}); raw audit: {raw}") from None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Capture CFBD data into a new immutable snapshot")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--season-type", choices=("regular", "postseason"), required=True)
    parser.add_argument("--week", type=int)
    parser.add_argument("--max-completed-games", type=int, default=100)
    parser.add_argument("--data-dir", type=Path, default=Paths().data_dir)
    args = parser.parse_args(argv)
    try:
        output = ingest_cfbd_batch(**vars(args))
    except (BatchIngestionError, ValueError, OSError) as exc:
        print(f"Ingestion did not publish a dataset: {exc}", file=sys.stderr)
        return 1
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    print(f"Published snapshot: {output}")
    print(f"Counts: {manifest['counts']}")
    print("Active model input CSVs were not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
