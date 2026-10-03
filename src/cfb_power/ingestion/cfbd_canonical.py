"""Pure CFBD-to-canonical mapping; no API calls or filesystem writes."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from cfb_power.schemas import GAME_COLUMNS, TEAM_GAME_STAT_COLUMNS
from cfb_power.validation import DataValidationError

PLAY_COUNT_BASIS = "passing_attempts_plus_rushing_attempts_proxy"


def _integer(value: Any, label: str, minimum: int | None = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise DataValidationError(f"{label}: expected an integer")
    if not re.fullmatch(r"-?\d+", str(value)):
        raise DataValidationError(f"{label}: malformed integer {value!r}")
    result = int(value)
    if minimum is not None and result < minimum:
        raise DataValidationError(f"{label}: must be at least {minimum}")
    return result


def _records(payload: Any, label: str) -> list[dict]:
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise DataValidationError(f"{label}: expected a list of objects")
    return payload


def _pair(value: Any, label: str, bounded: bool = True) -> tuple[int, int]:
    if not isinstance(value, str) or not re.fullmatch(r"\d+-\d+", value):
        raise DataValidationError(f"{label}: expected a count pair such as 7-16")
    first, second = map(int, value.split("-"))
    if bounded and first > second:
        raise DataValidationError(f"{label}: successes exceed attempts")
    return first, second


def _possession(value: Any) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"\d+:\d{2}", value):
        raise DataValidationError("possessionTime: expected minutes:seconds")
    minutes, seconds = map(int, value.split(":"))
    if seconds >= 60:
        raise DataValidationError("possessionTime: seconds must be below 60")
    return minutes * 60 + seconds


def normalize_games(payload: Any) -> pd.DataFrame:
    """Normalize an FBS-vs-FBS schedule batch; reject unsupported records."""
    rows, seen = [], set()
    for game in _records(payload, "games"):
        game_id = _integer(game.get("id"), "game id", 1)
        if game_id in seen:
            raise DataValidationError(f"duplicate game id: {game_id}")
        seen.add(game_id)
        if any(game.get(f"{side}Classification") != "fbs" for side in ("home", "away")):
            raise DataValidationError(f"game {game_id}: requires two FBS teams")
        if any(type(game.get(key)) is not bool for key in ("completed", "neutralSite")):
            raise DataValidationError(f"game {game_id}: completed/neutralSite must be booleans")
        if game.get("startTimeTBD") is True:
            raise DataValidationError(f"game {game_id}: kickoff time is unconfirmed")
        date = game.get("startDate")
        if not isinstance(date, str) or not re.search(r"(?:Z|[+-]\d{2}:\d{2})$", date):
            raise DataValidationError(f"game {game_id}: timezone-aware kickoff required")
        try:
            kickoff = pd.Timestamp(date)
        except (ValueError, TypeError) as exc:
            raise DataValidationError(f"game {game_id}: invalid kickoff") from exc
        home = _integer(game.get("homeId"), "homeId", 1)
        away = _integer(game.get("awayId"), "awayId", 1)
        if home == away:
            raise DataValidationError(f"game {game_id}: team playing itself")
        completed = game["completed"]
        scores = {}
        for side in ("home", "away"):
            value = game.get(f"{side}Points")
            if not completed and value is not None:
                raise DataValidationError(f"game {game_id}: unfinished game has a score")
            scores[side] = _integer(value, f"{side}Points") if completed else None
        rows.append({
            "game_id": game_id, "season": _integer(game.get("season"), "season", 1),
            "week": _integer(game.get("week"), "week"),
            "game_date": kickoff.tz_convert("UTC"),
            "home_team_id": home, "away_team_id": away,
            "home_score": scores["home"], "away_score": scores["away"],
            "neutral_site": game["neutralSite"], "completed": completed,
        })
    return pd.DataFrame(rows, columns=GAME_COLUMNS)


def _categories(team: dict) -> dict[str, Any]:
    values = {}
    for item in _records(team.get("stats"), "team stats"):
        category = item.get("category")
        if not isinstance(category, str) or not category:
            raise DataValidationError("invalid stat category")
        if category in values:
            raise DataValidationError(f"duplicate stat category: {category}")
        values[category] = item.get("stat")
    return values


def normalize_cfbd_canonical(
    games_payload: Any, team_stats_payload: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return canonical games and completed-game team stats as one validated batch.

    Red-zone columns remain missing. offensive_plays is a declared basic-box proxy.
    Every completed game must have exactly one box with two matching team records.
    """
    games = normalize_games(games_payload)
    schedule = {row["game_id"]: row for row in games.to_dict("records")}
    boxes = {}
    for box in _records(team_stats_payload, "team game boxes"):
        game_id = _integer(box.get("id"), "box game id", 1)
        if game_id in boxes:
            raise DataValidationError(f"duplicate box game id: {game_id}")
        if game_id not in schedule:
            raise DataValidationError(f"box references unknown game: {game_id}")
        if not schedule[game_id]["completed"]:
            raise DataValidationError(f"unfinished game has a box: {game_id}")
        boxes[game_id] = box
    rows = []
    for game_id, game in schedule.items():
        if not game["completed"]:
            continue
        if game_id not in boxes:
            raise DataValidationError(f"completed game missing box: {game_id}")
        teams = _records(boxes[game_id].get("teams"), "box teams")
        if len(teams) != 2:
            raise DataValidationError(f"game {game_id}: requires two team records")
        indexed = {}
        for team in teams:
            team_id = _integer(team.get("teamId"), "teamId", 1)
            if team_id in indexed:
                raise DataValidationError(f"duplicate team id: {team_id}")
            indexed[team_id] = team
        if set(indexed) != {game["home_team_id"], game["away_team_id"]}:
            raise DataValidationError(f"game {game_id}: team IDs disagree with schedule")
        stats = {team_id: _categories(team) for team_id, team in indexed.items()}
        turnovers = {team_id: _integer(value.get("turnovers"), "turnovers") for team_id, value in stats.items()}
        for side, opponent_side in (("home", "away"), ("away", "home")):
            team_id, opponent_id = game[f"{side}_team_id"], game[f"{opponent_side}_team_id"]
            team, values = indexed[team_id], stats[team_id]
            points = _integer(team.get("points"), "team points")
            if team.get("homeAway") != side or points != game[f"{side}_score"]:
                raise DataValidationError(f"game {game_id}: role or score mismatch")
            conversions, attempts = _pair(values.get("thirdDownEff"), "thirdDownEff")
            _, passing_attempts = _pair(values.get("completionAttempts"), "completionAttempts")
            penalties, penalty_yards = _pair(values.get("totalPenaltiesYards"), "totalPenaltiesYards", False)
            plays = passing_attempts + _integer(values.get("rushingAttempts"), "rushingAttempts")
            if plays <= 0:
                raise DataValidationError("offensive play-count proxy must be positive")
            total = _integer(values.get("totalYards"), "totalYards", None)
            passing = _integer(values.get("netPassingYards"), "netPassingYards", None)
            rushing = _integer(values.get("rushingYards"), "rushingYards", None)
            if total != passing + rushing:
                raise DataValidationError(f"game {game_id}: yardage components disagree")
            rows.append({
                "game_id": game_id, "team_id": team_id, "opponent_id": opponent_id,
                "is_home": side == "home", "points": points,
                "points_allowed": game[f"{opponent_side}_score"],
                "total_yards": total, "offensive_plays": plays,
                "passing_yards": passing, "rushing_yards": rushing,
                "turnovers": turnovers[team_id], "takeaways": turnovers[opponent_id],
                "third_down_attempts": attempts, "third_down_conversions": conversions,
                "red_zone_attempts": None, "red_zone_touchdowns": None,
                "penalties": penalties, "penalty_yards": penalty_yards,
                "time_of_possession_seconds": _possession(values.get("possessionTime")),
                "offensive_plays_basis": PLAY_COUNT_BASIS,
            })
    columns = [*TEAM_GAME_STAT_COLUMNS, "offensive_plays_basis"]
    frame = pd.DataFrame(rows, columns=columns)
    for column in ("red_zone_attempts", "red_zone_touchdowns"):
        frame[column] = frame[column].astype("float64")
    return games, frame
