"""Centralized helper for team name resolution and DataFrame enrichment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def resolve_team_names(data_dir: Path | str | None = None) -> dict[str | int, str]:
    """Resolve a mapping from team IDs (int and str) to display team names.

    Searches in order:
    1. Saved cache file (team_names.json) in processed dir
    2. Datasets manifest inputs (from latest verified dataset)
    3. Raw CFBD schedule json archives in raw/cfbd/
    """
    if data_dir is None:
        data_dir = Path("data")
    else:
        data_dir = Path(data_dir)

    names: dict[str | int, str] = {}

    # 1. Cached team_names.json in processed
    cached_path = data_dir / "processed" / "team_names.json"
    if cached_path.is_file():
        try:
            raw = json.loads(cached_path.read_text(encoding="utf-8"))
            for k, v in raw.items():
                if isinstance(v, str) and v:
                    names[k] = v
                    try:
                        names[int(k)] = v
                    except ValueError:
                        pass
            if len(names) > 50:
                return names
        except Exception:
            pass

    # 2. Check datasets manifests
    datasets_dir = data_dir / "datasets"
    if datasets_dir.is_dir():
        manifests = sorted(
            datasets_dir.glob("*/manifest.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for manifest_path in manifests:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for item in manifest.get("inputs", []):
                    snap_dir = Path(item["path"])
                    snap_manifest_path = snap_dir / "manifest.json"
                    if snap_manifest_path.is_file():
                        snap_manifest = json.loads(snap_manifest_path.read_text(encoding="utf-8"))
                        raw_archive = Path(snap_manifest.get("raw_archive", ""))
                        games_json = raw_archive / "games.json"
                        if games_json.is_file():
                            games = json.loads(games_json.read_text(encoding="utf-8"))
                            _extract_names_from_games(games, names)
                if len(names) > 50:
                    break
            except Exception:
                continue

    # 3. Direct scan of raw cfbd archives if still incomplete
    if len(names) < 50:
        raw_dir = data_dir / "raw" / "cfbd"
        if raw_dir.is_dir():
            for games_json in raw_dir.glob("*/*/games.json"):
                try:
                    games = json.loads(games_json.read_text(encoding="utf-8"))
                    _extract_names_from_games(games, names)
                except Exception:
                    continue

    # Cache if found substantial names
    if len(names) > 50 and cached_path.parent.is_dir():
        try:
            int_keys_only = {str(k): v for k, v in names.items() if isinstance(k, int)}
            cached_path.write_text(json.dumps(int_keys_only, indent=2), encoding="utf-8")
        except Exception:
            pass

    return names


def _extract_names_from_games(games: list[dict[str, Any]], names: dict[str | int, str]) -> None:
    for game in games:
        if not isinstance(game, dict):
            continue
        for side in ("home", "away"):
            tid = game.get(f"{side}Id")
            tname = game.get(f"{side}Team")
            if tid is not None and isinstance(tname, str) and tname:
                names[tid] = tname
                names[str(tid)] = tname
                try:
                    names[int(tid)] = tname
                except (ValueError, TypeError):
                    pass


def enrich_predictions_frame(
    predictions: pd.DataFrame, team_names: dict[str | int, str] | None = None
) -> pd.DataFrame:
    """Add friendly home_team, away_team, and matchup columns to predictions DataFrame."""
    df = predictions.copy()
    names = team_names or {}

    if "home_team" not in df.columns and "home_team_id" in df.columns:
        df["home_team"] = (
            df["home_team_id"]
            .map(names)
            .fillna(df["home_team_id"].astype(str))
            .replace("", "Unknown")
        )
    elif "home_team" in df.columns:
        df["home_team"] = df["home_team"].astype(object)
        missing_mask = df["home_team"].isna() | (df["home_team"] == "")
        if missing_mask.any() and "home_team_id" in df.columns:
            df.loc[missing_mask, "home_team"] = (
                df.loc[missing_mask, "home_team_id"]
                .map(names)
                .fillna(df.loc[missing_mask, "home_team_id"].astype(str))
            )

    if "away_team" not in df.columns and "away_team_id" in df.columns:
        df["away_team"] = (
            df["away_team_id"]
            .map(names)
            .fillna(df["away_team_id"].astype(str))
            .replace("", "Unknown")
        )
    elif "away_team" in df.columns:
        df["away_team"] = df["away_team"].astype(object)
        missing_mask = df["away_team"].isna() | (df["away_team"] == "")
        if missing_mask.any() and "away_team_id" in df.columns:
            df.loc[missing_mask, "away_team"] = (
                df.loc[missing_mask, "away_team_id"]
                .map(names)
                .fillna(df.loc[missing_mask, "away_team_id"].astype(str))
            )

    if "matchup" not in df.columns and "home_team" in df.columns and "away_team" in df.columns:
        if "neutral_site" in df.columns:
            is_neutral = df["neutral_site"].fillna(False).astype(bool)
            separator = is_neutral.map({True: " vs ", False: " @ "})
            suffix = is_neutral.map({True: " (N)", False: ""})
            df["matchup"] = (
                df["away_team"].astype(str) + separator + df["home_team"].astype(str) + suffix
            )
        else:
            df["matchup"] = df["away_team"].astype(str) + " @ " + df["home_team"].astype(str)

    # Reorder columns to put game context and team names up front
    front_cols = [
        "game_id",
        "season",
        "week",
        "game_date",
        "kickoff_utc",
        "matchup",
        "home_team",
        "away_team",
    ]
    present_front = [c for c in front_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in present_front]
    return df[present_front + other_cols]


def enrich_stats_frame(
    stats: pd.DataFrame, team_names: dict[str | int, str] | None = None
) -> pd.DataFrame:
    """Add friendly team_name and opponent_name to team_game_stats DataFrame."""
    df = stats.copy()
    names = team_names or {}

    if "team_name" not in df.columns and "team_id" in df.columns:
        df["team_name"] = (
            df["team_id"].map(names).fillna(df["team_id"].astype(str)).replace("", "Unknown")
        )
    if "opponent_name" not in df.columns and "opponent_id" in df.columns:
        df["opponent_name"] = (
            df["opponent_id"]
            .map(names)
            .fillna(df["opponent_id"].astype(str))
            .replace("", "Unknown")
        )

    front_cols = ["game_id", "game_date", "team_id", "team_name", "opponent_id", "opponent_name"]
    present_front = [c for c in front_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in present_front]
    return df[present_front + other_cols]
