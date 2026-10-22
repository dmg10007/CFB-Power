"""Normalize CFBD advanced box score payloads into one flat row per team."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

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


def aggregate_advanced_metrics(
    advanced_records: list[dict[str, Any]] | pd.DataFrame,
) -> dict[str | int, dict[str, float]]:
    """Aggregate game-level advanced box records into season-level team efficiency metrics."""
    if isinstance(advanced_records, list):
        df = pd.DataFrame(advanced_records)
    else:
        df = advanced_records.copy()

    if df.empty or "team" not in df.columns or "game_id" not in df.columns:
        return {}

    # Pair teams per game_id to compute allowed metrics
    by_game = df.groupby("game_id")
    enriched_rows: list[dict[str, Any]] = []
    for _, g in by_game:
        records = g.to_dict("records")
        if len(records) == 2:
            r0 = dict(records[0])
            r1 = dict(records[1])
            r0["opp_success_rate"] = r1.get("success_rates_overall_total")
            r0["opp_ppa"] = r1.get("ppa_overall_total")
            r0["opp_havoc"] = r1.get("havoc_total")
            r1["opp_success_rate"] = r0.get("success_rates_overall_total")
            r1["opp_ppa"] = r0.get("ppa_overall_total")
            r1["opp_havoc"] = r0.get("havoc_total")
            enriched_rows.extend([r0, r1])
        else:
            enriched_rows.extend(records)

    edf = pd.DataFrame(enriched_rows)
    metrics: dict[str | int, dict[str, float]] = {}
    for team, group in edf.groupby("team"):
        sr = (
            group["success_rates_overall_total"].dropna()
            if "success_rates_overall_total" in group
            else pd.Series()
        )
        opp_sr = group["opp_success_rate"].dropna() if "opp_success_rate" in group else pd.Series()
        ppa = group["ppa_overall_total"].dropna() if "ppa_overall_total" in group else pd.Series()
        opp_ppa = group["opp_ppa"].dropna() if "opp_ppa" in group else pd.Series()
        exp = (
            group["explosiveness_overall_total"].dropna()
            if "explosiveness_overall_total" in group
            else pd.Series()
        )
        havoc = group["havoc_total"].dropna() if "havoc_total" in group else pd.Series()
        ppo = (
            group["scoring_opportunities_points_per_opportunity"].dropna()
            if "scoring_opportunities_points_per_opportunity" in group
            else pd.Series()
        )

        metrics[team] = {
            "success_rate": float(sr.mean()) if not sr.empty else 0.0,
            "success_rate_allowed": float(opp_sr.mean()) if not opp_sr.empty else 0.0,
            "ppa_per_play": float(ppa.mean()) if not ppa.empty else 0.0,
            "ppa_per_play_allowed": float(opp_ppa.mean()) if not opp_ppa.empty else 0.0,
            "explosiveness": float(exp.mean()) if not exp.empty else 0.0,
            "havoc_rate": float(havoc.mean()) if not havoc.empty else 0.0,
            "points_per_opportunity": float(ppo.mean()) if not ppo.empty else 0.0,
        }
    return metrics
