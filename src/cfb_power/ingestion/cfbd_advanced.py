"""Normalize CFBD advanced box score payloads into one flat row per team."""

from __future__ import annotations

import re
from typing import Any

from cfb_power.advanced_schema import AdvancedSchemaError

_CAMEL = re.compile(r"(?<!^)(?=[A-Z])")


def _snake(name: str) -> str:
    return _CAMEL.sub("_", name).lower()


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, inner in value.items():
            _flatten(f"{prefix}_{_snake(str(key))}", inner, out)
    elif value is None or isinstance(value, (int, float, str, bool)):
        out[prefix] = value


def normalize_advanced_box(payload: Any, game_id: int) -> list[dict[str, Any]]:
    """Return one flat dict per team, merged across all category lists.

    Expected shape: {"teams": {category: [{"team": name, ...}, ...]}}.
    """
    teams = payload.get("teams") if isinstance(payload, dict) else None
    if not isinstance(teams, dict):
        raise AdvancedSchemaError("payload must contain a 'teams' object")

    rows: dict[str, dict[str, Any]] = {}
    for category, entries in teams.items():
        if not isinstance(entries, list):
            raise AdvancedSchemaError(f"teams.{category} must be a list")
        prefix = _snake(str(category))
        for entry in entries:
            name = entry.get("team") if isinstance(entry, dict) else None
            if not name:
                raise AdvancedSchemaError(f"teams.{category} entry missing 'team'")
            row = rows.setdefault(name, {"game_id": game_id, "team": name})
            for key, value in entry.items():
                if key != "team":
                    _flatten(f"{prefix}_{_snake(str(key))}", value, row)
    return list(rows.values())
