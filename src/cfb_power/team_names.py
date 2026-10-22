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
        except (OSError, json.JSONDecodeError, TypeError):
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
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                continue

    # 3. Direct scan of raw cfbd archives if still incomplete
    if len(names) < 50:
        raw_dir = data_dir / "raw" / "cfbd"
        if raw_dir.is_dir():
            for games_json in raw_dir.glob("*/*/games.json"):
                try:
                    games = json.loads(games_json.read_text(encoding="utf-8"))
                    _extract_names_from_games(games, names)
                except (OSError, json.JSONDecodeError, KeyError, TypeError):
                    continue

    # Cache if found substantial names
    if len(names) > 50 and cached_path.parent.is_dir():
        try:
            int_keys_only = {str(k): v for k, v in names.items() if isinstance(k, int)}
            cached_path.write_text(json.dumps(int_keys_only, indent=2), encoding="utf-8")
        except (OSError, ValueError, TypeError):
            pass

    return names


# Standard FBS Conference Fallback Mapping for major FBS teams
FBS_CONFERENCE_DEFAULTS: dict[str, str] = {
    # SEC
    "Alabama": "SEC",
    "Arkansas": "SEC",
    "Auburn": "SEC",
    "Florida": "SEC",
    "Georgia": "SEC",
    "Kentucky": "SEC",
    "LSU": "SEC",
    "Mississippi State": "SEC",
    "Missouri": "SEC",
    "Oklahoma": "SEC",
    "Ole Miss": "SEC",
    "South Carolina": "SEC",
    "Tennessee": "SEC",
    "Texas": "SEC",
    "Texas A&M": "SEC",
    "Vanderbilt": "SEC",
    # Big Ten
    "Illinois": "Big Ten",
    "Indiana": "Big Ten",
    "Iowa": "Big Ten",
    "Maryland": "Big Ten",
    "Michigan": "Big Ten",
    "Michigan State": "Big Ten",
    "Minnesota": "Big Ten",
    "Nebraska": "Big Ten",
    "Northwestern": "Big Ten",
    "Ohio State": "Big Ten",
    "Oregon": "Big Ten",
    "Penn State": "Big Ten",
    "Purdue": "Big Ten",
    "Rutgers": "Big Ten",
    "UCLA": "Big Ten",
    "USC": "Big Ten",
    "Washington": "Big Ten",
    "Wisconsin": "Big Ten",
    # Big 12
    "Arizona": "Big 12",
    "Arizona State": "Big 12",
    "Baylor": "Big 12",
    "BYU": "Big 12",
    "Cincinnati": "Big 12",
    "Colorado": "Big 12",
    "Houston": "Big 12",
    "Iowa State": "Big 12",
    "Kansas": "Big 12",
    "Kansas State": "Big 12",
    "Oklahoma State": "Big 12",
    "TCU": "Big 12",
    "Texas Tech": "Big 12",
    "UCF": "Big 12",
    "Utah": "Big 12",
    "West Virginia": "Big 12",
    # ACC
    "Boston College": "ACC",
    "California": "ACC",
    "Clemson": "ACC",
    "Duke": "ACC",
    "Florida State": "ACC",
    "Georgia Tech": "ACC",
    "Louisville": "ACC",
    "Miami": "ACC",
    "NC State": "ACC",
    "North Carolina": "ACC",
    "Pittsburgh": "ACC",
    "SMU": "ACC",
    "Stanford": "ACC",
    "Syracuse": "ACC",
    "Virginia": "ACC",
    "Virginia Tech": "ACC",
    "Wake Forest": "ACC",
    # Independents
    "Notre Dame": "FBS Independents",
    "UConn": "FBS Independents",
    "UMass": "FBS Independents",
    # Mountain West / Pac-12
    "Boise State": "Mountain West",
    "Colorado State": "Mountain West",
    "Fresno State": "Mountain West",
    "Oregon State": "Pac-12",
    "San Diego State": "Mountain West",
    "San José State": "Mountain West",
    "UNLV": "Mountain West",
    "Washington State": "Pac-12",
    # American
    "Army": "American Athletic",
    "Memphis": "American Athletic",
    "Navy": "American Athletic",
    "South Florida": "American Athletic",
    "Tulane": "American Athletic",
    "UTSA": "American Athletic",
    # Sun Belt
    "Appalachian State": "Sun Belt",
    "Coastal Carolina": "Sun Belt",
    "James Madison": "Sun Belt",
    "Louisiana": "Sun Belt",
    "Texas State": "Sun Belt",
    # MAC
    "Miami (OH)": "Mid-American",
    "Northern Illinois": "Mid-American",
    "Toledo": "Mid-American",
    # C-USA
    "Jacksonville State": "Conference USA",
    "Liberty": "Conference USA",
    "Western Kentucky": "Conference USA",
}


def resolve_team_conferences(data_dir: Path | str | None = None) -> dict[str | int, str]:
    """Resolve a mapping from team IDs and names to conference affiliations.

    Searches in order:
    1. Saved cache file (team_conferences.json) in processed dir
    2. Datasets manifest inputs (from latest verified dataset)
    3. Raw CFBD schedule json archives in raw/cfbd/
    4. Built-in FBS conference defaults
    """
    if data_dir is None:
        data_dir = Path("data")
    else:
        data_dir = Path(data_dir)

    conferences: dict[str | int, str] = dict(FBS_CONFERENCE_DEFAULTS)

    # 1. Cached team_conferences.json in processed
    cached_path = data_dir / "processed" / "team_conferences.json"
    if cached_path.is_file():
        try:
            raw = json.loads(cached_path.read_text(encoding="utf-8"))
            for k, v in raw.items():
                if isinstance(v, str) and v:
                    conferences[k] = v
                    try:
                        conferences[int(k)] = v
                    except ValueError:
                        pass
            if len(conferences) > 50:
                return conferences
        except (OSError, json.JSONDecodeError, TypeError):
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
                            _extract_conferences_from_games(games, conferences)
                if len(conferences) > 50:
                    break
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                continue

    # 3. Direct scan of raw cfbd archives if still incomplete
    if len(conferences) < 50:
        raw_dir = data_dir / "raw" / "cfbd"
        if raw_dir.is_dir():
            for games_json in raw_dir.glob("*/*/games.json"):
                try:
                    games = json.loads(games_json.read_text(encoding="utf-8"))
                    _extract_conferences_from_games(games, conferences)
                except (OSError, json.JSONDecodeError, KeyError, TypeError):
                    continue

    # Cache if found substantial mappings
    if len(conferences) > 50 and cached_path.parent.is_dir():
        try:
            cached_path.write_text(json.dumps(conferences, indent=2), encoding="utf-8")
        except (OSError, ValueError, TypeError):
            pass

    return conferences


def _extract_conferences_from_games(
    games: list[dict[str, Any]], conferences: dict[str | int, str]
) -> None:
    for game in games:
        if not isinstance(game, dict):
            continue
        for side in ("home", "away"):
            tid = game.get(f"{side}Id")
            tname = game.get(f"{side}Team")
            conf = game.get(f"{side}Conference")
            if isinstance(conf, str) and conf.strip():
                clean_conf = conf.strip()
                if tid is not None:
                    conferences[tid] = clean_conf
                    conferences[str(tid)] = clean_conf
                    try:
                        conferences[int(tid)] = clean_conf
                    except (ValueError, TypeError):
                        pass
                if isinstance(tname, str) and tname.strip():
                    conferences[tname.strip()] = clean_conf


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
