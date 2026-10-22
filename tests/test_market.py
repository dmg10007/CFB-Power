"""Tests for betting market lines parsing, edge calculation, and ATS evaluation."""

import pandas as pd
import pytest

from cfb_power.market import (
    compute_market_edges,
    evaluate_market_performance,
    parse_consensus_lines,
)


def test_parse_consensus_lines_extracts_preferred_lines():
    payload = [
        {
            "id": 1001,
            "homeTeam": "Georgia",
            "awayTeam": "Alabama",
            "lines": [
                {"provider": "DraftKings", "spread": -3.0, "overUnder": 51.5},
                {"provider": "consensus", "spread": -3.5, "overUnder": 52.0},
            ],
        },
        {
            "id": 1002,
            "homeTeam": "Ohio State",
            "awayTeam": "Michigan",
            "lines": [
                {"provider": "Bovada", "spread": -7.0, "overUnder": 48.0},
            ],
        },
    ]

    parsed = parse_consensus_lines(payload)
    assert 1001 in parsed
    assert parsed[1001]["market_spread"] == -3.5
    assert parsed[1001]["market_total"] == 52.0
    assert parsed[1001]["provider"] == "consensus"

    assert ("georgia", "alabama") in parsed
    assert parsed[("georgia", "alabama")]["market_spread"] == -3.5

    assert 1002 in parsed
    assert parsed[1002]["market_spread"] == -7.0
    assert parsed[1002]["provider"] == "Bovada"


def test_compute_market_edges_home_favorite():
    # Model: Home 31, Away 24 -> Model Home Spread = -7.0 (Home favored by 7), Total = 55.0
    # Market: Home -3.5, Total = 50.0
    # Spread Edge on Home = -3.5 - (-7.0) = +3.5 pts
    # Total Edge on Over = 55.0 - 50.0 = +5.0 pts
    edges = compute_market_edges(
        projected_home_points=31.0,
        projected_away_points=24.0,
        market_spread=-3.5,
        market_total=50.0,
    )
    assert edges["model_home_spread"] == -7.0
    assert edges["spread_edge"] == 3.5
    assert edges["model_total"] == 55.0
    assert edges["total_edge"] == 5.0
    assert edges["value_tier"] == "Strong Value"
    assert any("Home Spread Edge" in note for note in edges["edge_notes"])
    assert any("OVER Edge" in note for note in edges["edge_notes"])


def test_compute_market_edges_away_value():
    # Model: Home 21, Away 24 -> Model Home Spread = +3.0 (Away favored by 3)
    # Market: Home -4.0 (Market favors Home)
    # Spread Edge = -4.0 - (+3.0) = -7.0 pts (Strong edge on Away)
    edges = compute_market_edges(
        projected_home_points=21.0,
        projected_away_points=24.0,
        market_spread=-4.0,
        market_total=45.0,
    )
    assert edges["spread_edge"] == -7.0
    assert edges["value_tier"] == "Strong Value"
    assert any("Away Spread Edge" in note for note in edges["edge_notes"])


def test_evaluate_market_performance_ats_and_totals():
    preds = pd.DataFrame(
        [
            # Game 1: Market Home -3.5, Model edge +3.5 on Home.
            # Final: Home 30, Away 20 (Margin +10 -> Home covers -3.5 -> Model win)
            {
                "game_id": 1,
                "market_spread": -3.5,
                "spread_edge": 3.5,
                "market_total": 48.0,
                "total_edge": 4.0,
            },
            # Game 2: Market Home -7.0, Model edge -4.0 on Away.
            # Final: Home 24, Away 20 (Margin +4 -> Away covers +7.0 -> Model win)
            {
                "game_id": 2,
                "market_spread": -7.0,
                "spread_edge": -4.0,
                "market_total": 55.0,
                "total_edge": -3.0,
            },
            # Game 3: Market Home -10.0, Model edge +3.0 on Home.
            # Final: Home 24, Away 21 (Margin +3 -> Away covers -> Model loss)
            {
                "game_id": 3,
                "market_spread": -10.0,
                "spread_edge": 3.0,
                "market_total": 50.0,
                "total_edge": -2.0,
            },
        ]
    )

    actuals = pd.DataFrame(
        [
            {"game_id": 1, "home_score": 30, "away_score": 20, "completed": True},
            {"game_id": 2, "home_score": 24, "away_score": 20, "completed": True},
            {"game_id": 3, "home_score": 24, "away_score": 21, "completed": True},
        ]
    )

    eval_res = evaluate_market_performance(preds, actuals)
    # 2 wins, 1 loss -> 2-1-0 (66.7%)
    assert eval_res["ats_record"] == "2-1-0"
    assert eval_res["ats_win_pct"] == pytest.approx(0.667, abs=1e-2)
    assert eval_res["ats_games"] == 3
