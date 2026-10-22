"""Explicit, hash-bound selection of whole-game snapshot versions."""

import hashlib
import json
from pathlib import Path


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class ReviewRequired(RuntimeError):
    def __init__(self, plan):
        super().__init__("Snapshot revisions require review before assembly.")
        self.plan = plan


def make_review(loaded):
    inputs, versions = [], {}
    for manifest, path, games, stats in loaded:
        source = {
            "run_id": manifest["run_id"],
            "path": str(path),
            "manifest_sha256": hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest(),
            "captured_at": manifest["started_at_utc"],
        }
        inputs.append(source)
        for (game_id,), game in games.items():
            bundle = {
                "game": game,
                "teams": sorted(
                    [row for (gid, _), row in stats.items() if gid == game_id],
                    key=lambda row: row["team_id"],
                ),
            }
            versions.setdefault(str(game_id), []).append({**source, "bundle": bundle})
    conflicts = {}
    for game_id, candidates in versions.items():
        complete = [v for v in candidates if v["bundle"]["game"]["completed"]]
        eligible = complete or candidates
        if len({_digest(v["bundle"]) for v in eligible}) > 1:
            conflicts[game_id] = eligible
    return {
        "token": _digest({"inputs": inputs, "versions": versions}),
        "inputs": inputs,
        "conflicts": conflicts,
    }


def apply_review(loaded, decisions, token, check):
    """Return merge inputs and audit, without changing source dictionaries."""
    plan = make_review(loaded)
    if token != plan["token"]:
        raise ValueError("Snapshot inputs changed; review them again.")
    if set(decisions) != set(plan["conflicts"]):
        raise ValueError("Select exactly one version for every conflicting game.")
    selections = {}
    for game_id, candidates in plan["conflicts"].items():
        identity = {
            (
                v["bundle"]["game"]["home_team_id"],
                v["bundle"]["game"]["away_team_id"],
                v["bundle"]["game"]["season"],
            )
            for v in candidates
        }
        if len(identity) != 1:
            raise ValueError(
                f"Game {game_id} changes team identities or season; manual investigation required."
            )
        selected = next((v for v in candidates if v["run_id"] == decisions[game_id]), None)
        if selected is None:
            raise ValueError(f"Selected snapshot is not a candidate for game {game_id}.")
        bundle = selected["bundle"]
        if bundle["game"]["completed"]:
            check(
                {(int(game_id),): bundle["game"]},
                {(int(game_id), row["team_id"]): row for row in bundle["teams"]},
            )
        elif bundle["teams"]:
            raise ValueError("An upcoming game must not have team statistics.")
        selections[int(game_id)] = selected
    filtered = []
    for manifest, path, games, stats in loaded:
        run_id = manifest["run_id"]

        def keep(gid, target_run_id=run_id):
            return gid not in selections or selections[gid]["run_id"] == target_run_id

        filtered.append(
            (
                manifest,
                path,
                {k: v for k, v in games.items() if keep(k[0])},
                {k: v for k, v in stats.items() if keep(k[0])},
            )
        )
    return filtered, {
        "review_token": token,
        "decisions": dict(decisions),
        "reviewed_conflicts": plan["conflicts"],
    }


def inspect_snapshots(snapshot_dirs):
    from .dataset_workflow import SNAPSHOT_SCHEMA, _index, _load_verified
    from .ingestion.cfbd_canonical import PLAY_COUNT_BASIS

    loaded = []
    seen = set()
    for item in snapshot_dirs:
        path = Path(item).resolve()
        before = hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
        manifest, games, stats = _load_verified(path, SNAPSHOT_SCHEMA, path.name)
        if before != hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest():
            raise ValueError("Snapshot changed while being verified.")
        if manifest.get("offensive_plays_basis") != PLAY_COUNT_BASIS:
            raise ValueError("Unsupported offensive-play basis.")
        if manifest["run_id"] in seen:
            raise ValueError("Duplicate snapshot run ID.")
        seen.add(manifest["run_id"])
        loaded.append(
            (
                manifest,
                path,
                _index(games, ("game_id",), path.name),
                _index(stats, ("game_id", "team_id"), path.name),
            )
        )
    loaded.sort(key=lambda v: (v[0]["started_at_utc"], v[0]["run_id"]))
    if not loaded:
        raise ValueError("No published snapshots found.")
    return make_review(loaded)


def assemble_or_review(snapshot_dirs, data_dir):
    from .dataset_workflow import assemble_dataset

    paths = list(snapshot_dirs)
    plan = inspect_snapshots(paths)
    if plan["conflicts"]:
        raise ReviewRequired(plan)
    return assemble_dataset(paths, data_dir)


def resume_review(data_dir, decisions, token):
    """No client or capture call: resume using published snapshots."""
    from .dataset_workflow import assemble_dataset, discover_snapshots
    from .upcoming import predict_upcoming

    dataset = assemble_dataset(
        discover_snapshots(data_dir), data_dir, revision_decisions=decisions, review_token=token
    )
    return predict_upcoming(dataset, data_dir)
