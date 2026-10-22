"""Betting market lines parser, edge calculator, and ATS / Total performance evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Priority order for preferred lines providers
PREFERRED_PROVIDERS = ("consensus", "DraftKings", "ESPN Bet", "Bovada", "Caesars", "BetMGM")


def parse_consensus_lines(
    lines_payload: list[dict[str, Any]],
) -> dict[str | int | tuple[str, str], dict[str, Any]]:
    """Parse CFBD /lines response into a clean dictionary keyed by game_id and (home, away)."""
    result: dict[str | int | tuple[str, str], dict[str, Any]] = {}
    if not isinstance(lines_payload, list):
        return result

    for item in lines_payload:
        if not isinstance(item, dict):
            continue
        game_id = item.get("id")
        home = str(item.get("homeTeam") or "").strip()
        away = str(item.get("awayTeam") or "").strip()
        lines_list = item.get("lines") or []

        if not isinstance(lines_list, list) or not lines_list:
            continue

        # Find best line provider
        chosen_line: dict[str, Any] | None = None
        for pref in PREFERRED_PROVIDERS:
            for line in lines_list:
                if isinstance(line, dict) and str(line.get("provider", "")).lower() == pref.lower():
                    if line.get("spread") is not None or line.get("overUnder") is not None:
                        chosen_line = line
                        break
            if chosen_line is not None:
                break

        if chosen_line is None:
            for line in lines_list:
                if isinstance(line, dict) and (
                    line.get("spread") is not None or line.get("overUnder") is not None
                ):
                    chosen_line = line
                    break

        if chosen_line is None:
            continue

        raw_spread = chosen_line.get("spread")
        raw_ou = chosen_line.get("overUnder")

        try:
            market_spread = float(raw_spread) if raw_spread is not None else None
        except (ValueError, TypeError):
            market_spread = None

        try:
            market_total = float(raw_ou) if raw_ou is not None else None
        except (ValueError, TypeError):
            market_total = None

        line_info: dict[str, Any] = {
            "game_id": game_id,
            "home_team": home,
            "away_team": away,
            "market_spread": market_spread,
            "market_total": market_total,
            "provider": chosen_line.get("provider", "market"),
            "formatted_spread": chosen_line.get("formattedSpread"),
            "home_moneyline": chosen_line.get("homeMoneyline"),
            "away_moneyline": chosen_line.get("awayMoneyline"),
        }

        if game_id is not None:
            result[game_id] = line_info
            result[str(game_id)] = line_info
        if home and away:
            result[(home.lower(), away.lower())] = line_info

    return result


def compute_market_edges(
    projected_home_points: float,
    projected_away_points: float,
    market_spread: float | None,
    market_total: float | None,
) -> dict[str, Any]:
    """Compute spread and total edges against betting market lines.

    Convention:
    - market_spread: home point spread (e.g. -6.5 means Home is favored by 6.5)
    - model_home_spread: projected_away - projected_home (e.g. 24 - 31 = -7.0 means Home favored by 7.0)
    - spread_edge = market_spread - model_home_spread:
        * Positive edge (> 0) means value is on Home covering (e.g. -3.5 - (-7.0) = +3.5 pts on Home)
        * Negative edge (< 0) means value is on Away covering (e.g. -10.0 - (-7.0) = -3.0 pts on Away)
    - total_edge = model_total - market_total:
        * Positive edge (> 0) means value is on the OVER
        * Negative edge (< 0) means value is on the UNDER
    """
    model_home_spread = round(projected_away_points - projected_home_points, 1)
    model_total = round(projected_home_points + projected_away_points, 1)

    spread_edge = None
    if market_spread is not None:
        spread_edge = round(market_spread - model_home_spread, 1)

    total_edge = None
    if market_total is not None:
        total_edge = round(model_total - market_total, 1)

    # Determine value tier
    abs_sp = abs(spread_edge) if spread_edge is not None else 0.0
    abs_tot = abs(total_edge) if total_edge is not None else 0.0

    if abs_sp >= 3.5 or abs_tot >= 4.0:
        value_tier = "Strong Value"
    elif abs_sp >= 2.0 or abs_tot >= 2.5:
        value_tier = "Moderate Value"
    else:
        value_tier = "Fair / Aligned"

    # Edge summary string
    edge_notes: list[str] = []
    if spread_edge is not None and abs_sp >= 1.5:
        side = "Home" if spread_edge > 0 else "Away"
        edge_notes.append(f"{side} Spread Edge: {abs_sp:+.1f} pts")
    if total_edge is not None and abs_tot >= 2.0:
        over_under = "OVER" if total_edge > 0 else "UNDER"
        edge_notes.append(f"{over_under} Edge: {abs_tot:+.1f} pts")

    return {
        "model_home_spread": model_home_spread,
        "market_spread": market_spread,
        "spread_edge": spread_edge,
        "model_total": model_total,
        "market_total": market_total,
        "total_edge": total_edge,
        "value_tier": value_tier,
        "edge_notes": edge_notes,
    }


def evaluate_market_performance(
    predictions_df: pd.DataFrame,
    actual_games_df: pd.DataFrame,
) -> dict[str, Any]:
    """Grade pregame forecast edges against actual final scores and market lines."""
    merged = predictions_df.merge(
        actual_games_df[["game_id", "home_score", "away_score", "completed"]],
        on="game_id",
        how="inner",
    )
    completed = merged[merged["completed"].astype(bool)].copy()
    if completed.empty or "market_spread" not in completed.columns:
        return {}

    # Valid lines only
    with_spread = completed.dropna(subset=["market_spread"]).copy()
    ats_results: list[str] = []
    for _, row in with_spread.iterrows():
        actual_home_margin = row["home_score"] - row["away_score"]
        # In betting: home covers if actual_home_margin + market_spread > 0
        # (e.g. actual margin +10, spread -6.5 -> +3.5 > 0 => Home covered)
        line = row["market_spread"]
        edge = row.get("spread_edge")

        if edge is None or np.isnan(edge) or abs(edge) < 0.5:
            continue

        model_picked_home = edge > 0
        cover_metric = actual_home_margin + line
        if abs(cover_metric) < 0.1:
            ats_results.append("push")
        elif (cover_metric > 0 and model_picked_home) or (
            cover_metric < 0 and not model_picked_home
        ):
            ats_results.append("win")
        else:
            ats_results.append("loss")

    ats_wins = ats_results.count("win")
    ats_losses = ats_results.count("loss")
    ats_pushes = ats_results.count("push")
    ats_total = ats_wins + ats_losses
    ats_win_pct = round(ats_wins / ats_total, 3) if ats_total > 0 else 0.0

    # Over / Under evaluation
    ou_results: list[str] = []
    if "market_total" in completed.columns:
        with_total = completed.dropna(subset=["market_total"]).copy()
        for _, row in with_total.iterrows():
            actual_total = row["home_score"] + row["away_score"]
            line = row["market_total"]
            edge = row.get("total_edge")
            if edge is None or np.isnan(edge) or abs(edge) < 1.0:
                continue

            model_picked_over = edge > 0
            if abs(actual_total - line) < 0.1:
                ou_results.append("push")
            elif (actual_total > line and model_picked_over) or (
                actual_total < line and not model_picked_over
            ):
                ou_results.append("win")
            else:
                ou_results.append("loss")

    ou_wins = ou_results.count("win")
    ou_losses = ou_results.count("loss")
    ou_pushes = ou_results.count("push")
    ou_total = ou_wins + ou_losses
    ou_win_pct = round(ou_wins / ou_total, 3) if ou_total > 0 else 0.0

    return {
        "ats_record": f"{ats_wins}-{ats_losses}-{ats_pushes}",
        "ats_win_pct": ats_win_pct,
        "ats_games": ats_total,
        "ou_record": f"{ou_wins}-{ou_losses}-{ou_pushes}",
        "ou_win_pct": ou_win_pct,
        "ou_games": ou_total,
    }
