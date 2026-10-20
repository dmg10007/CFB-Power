from pathlib import Path

from cfb_power.dashboard_power import get_live_power_rankings


def test_get_live_power_rankings_with_active_data():
    rankings_obj, df, names = get_live_power_rankings(Path("data"))
    assert rankings_obj is not None
    assert df is not None
    assert len(df) > 50
    assert "team_name" in df.columns
    assert "power_rating" in df.columns
    assert "tier" in df.columns
    assert df["team_name"].iloc[0] != ""
