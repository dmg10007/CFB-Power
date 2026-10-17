"""Development-only CFBD schema probe; never publishes model inputs."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import dotenv_values
from .config import PROJECT_ROOT
from .ingestion.cfbd_client import CfbdClient, CfbdHttpError


def probe(*, season, season_type="regular", game_id=None, root=PROJECT_ROOT, client=None):
    if type(season) is not int or season < 1:
        raise ValueError("season must be a positive integer")
    if season_type not in ("regular", "postseason"):
        raise ValueError("season_type must be regular or postseason")
    if game_id is not None and (type(game_id) is not int or game_id <= 0):
        raise ValueError("game_id must be a positive integer")
    root = Path(root).resolve()
    if client is None:
        key = dotenv_values(root / ".env", interpolate=False).get("CFBD_API_KEY") or os.getenv("CFBD_API_KEY")
        if not key:
            raise ValueError("CFBD_API_KEY is missing; save it in the project .env or export it.")
        client = CfbdClient(api_key=key)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex
    output = root / "data" / "probes" / "cfbd" / run_id
    output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    games = client.get_json("/games", {"year": season, "seasonType": season_type, "classification": "fbs"})
    if not isinstance(games, list) or any(not isinstance(game, dict) for game in games):
        raise ValueError("Schedule must be a list of game objects.")
    save(f"games_{season}.json", games)
    eligible = [game for game in games if game.get("completed") is True
                and game.get("homeClassification") == "fbs" and game.get("awayClassification") == "fbs"
                and game.get("season") == season and game.get("seasonType") == season_type
                and type(game.get("id")) is int and game["id"] > 0
                and (game_id is None or game["id"] == game_id)]
    if not eligible:
        raise ValueError("No matching completed FBS-vs-FBS game in the requested scope.")
    game = sorted(eligible, key=lambda row: row["id"])[0]
    selected_id = game["id"]
    boxes = client.get_json("/games/teams", {"id": selected_id})
    if (not isinstance(boxes, list) or len(boxes) != 1 or not isinstance(boxes[0], dict)
            or boxes[0].get("id") != selected_id):
        raise ValueError("Box request must return exactly one matching game.")
    teams = boxes[0].get("teams")
    if (not isinstance(teams, list) or len(teams) != 2
            or any(not isinstance(team, dict) for team in teams)):
        raise ValueError("Box score must contain two team objects.")
    expected = {game.get("homeId"), game.get("awayId")}
    if (len(expected) != 2 or any(type(team_id) is not int or team_id <= 0 for team_id in expected)
            or {team.get("teamId") for team in teams} != expected):
        raise ValueError("Box-score teams do not match the schedule.")
    for team in teams:
        stats = team.get("stats")
        if not isinstance(stats, list) or any(not isinstance(stat, dict) for stat in stats):
            raise ValueError("Each team must have a list of stat objects.")
        fields = sorted({key for stat in stats for key in stat})
        print(f"Team {team['teamId']}: {len(stats)} stats; fields: {', '.join(fields)}")
    save(f"team_stats_{selected_id}.json", boxes)
    advanced = client.get_json("/game/box/advanced", {"id": selected_id})
    save(f"advanced_{selected_id}.json", advanced)
    print(f"Probe game ID: {selected_id}")
    print(f"Advanced-box payload type: {type(advanced).__name__}")
    if isinstance(advanced, dict):
        print("Advanced-box fields: " + ", ".join(sorted(advanced)))
    print(f"Saved local probe payloads in: {output}")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect CFBD schemas; writes only local probe outputs")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--season-type", choices=("regular", "postseason"), default="regular")
    parser.add_argument("--game-id", type=int)
    args = parser.parse_args(argv)
    try:
        probe(**vars(args))
    except CfbdHttpError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError:
        print("Probe failed: check key configuration, arguments, and response schema.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Probe failed ({type(exc).__name__}); no response body displayed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
