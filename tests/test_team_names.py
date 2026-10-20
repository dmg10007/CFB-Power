import pandas as pd

from cfb_power.team_names import (
    enrich_predictions_frame,
    enrich_stats_frame,
    resolve_team_names,
)


def test_resolve_team_names_finds_active_names():
    names = resolve_team_names("data")
    assert isinstance(names, dict)
    assert len(names) > 50
    # Both int and string keys should be present
    sample_key = next(k for k in names.keys() if isinstance(k, int))
    assert isinstance(names[sample_key], str)
    assert names[str(sample_key)] == names[sample_key]


def test_enrich_predictions_frame():
    df = pd.DataFrame(
        {
            "game_id": [1],
            "season": [2026],
            "week": [1],
            "home_team_id": [10],
            "away_team_id": [20],
            "neutral_site": [False],
            "home_expected_points": [28.0],
            "away_expected_points": [21.0],
        }
    )
    names = {10: "Ohio State", 20: "Michigan"}
    enriched = enrich_predictions_frame(df, names)

    assert "matchup" in enriched.columns
    assert "home_team" in enriched.columns
    assert "away_team" in enriched.columns
    assert enriched["home_team"].iloc[0] == "Ohio State"
    assert enriched["away_team"].iloc[0] == "Michigan"
    assert enriched["matchup"].iloc[0] == "Michigan @ Ohio State"
    # Column ordering puts game context and names first
    assert list(enriched.columns[:5]) == ["game_id", "season", "week", "matchup", "home_team"]


def test_enrich_predictions_frame_neutral_site():
    df = pd.DataFrame(
        {
            "game_id": [2],
            "home_team_id": [10],
            "away_team_id": [20],
            "neutral_site": [True],
        }
    )
    names = {10: "Georgia", 20: "Alabama"}
    enriched = enrich_predictions_frame(df, names)
    assert enriched["matchup"].iloc[0] == "Alabama vs Georgia (N)"


def test_enrich_stats_frame():
    df = pd.DataFrame(
        {
            "game_id": [1],
            "game_date": ["2026-09-01"],
            "team_id": [10],
            "opponent_id": [20],
            "points": [35],
        }
    )
    names = {10: "Ohio State", 20: "Michigan"}
    enriched = enrich_stats_frame(df, names)

    assert "team_name" in enriched.columns
    assert "opponent_name" in enriched.columns
    assert enriched["team_name"].iloc[0] == "Ohio State"
    assert enriched["opponent_name"].iloc[0] == "Michigan"
    assert list(enriched.columns[:6]) == [
        "game_id",
        "game_date",
        "team_id",
        "team_name",
        "opponent_id",
        "opponent_name",
    ]
