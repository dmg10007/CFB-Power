import json
from pathlib import Path

import pytest

from cfb_power.advanced_schema import AdvancedSchemaError
from cfb_power.ingestion.cfbd_advanced import normalize_advanced_box

FIXTURE = Path(__file__).parent / "fixtures" / "cfbd" / "advanced_box_verified_shape.json"


@pytest.fixture()
def payload():
    return json.loads(FIXTURE.read_text())


def test_one_row_per_team(payload):
    rows = normalize_advanced_box(payload, game_id=1)
    assert sorted(r["team"] for r in rows) == ["Alpha", "Beta"]


def test_flattens_nested_values(payload):
    alpha = {r["team"]: r for r in normalize_advanced_box(payload, 1)}["Alpha"]
    assert alpha["ppa_plays"] == 60
    assert alpha["ppa_overall_total"] == 0.2
    assert alpha["ppa_overall_quarter1"] == 0.1


def test_merges_across_categories(payload):
    alpha = {r["team"]: r for r in normalize_advanced_box(payload, 1)}["Alpha"]
    assert alpha["success_rates_overall_total"] == 0.45
    assert alpha["havoc_total"] == 0.1
    assert alpha["game_id"] == 1


def test_missing_teams_raises():
    with pytest.raises(AdvancedSchemaError):
        normalize_advanced_box({"gameInfo": {}}, 1)


def test_entry_without_team_name_raises():
    with pytest.raises(AdvancedSchemaError):
        normalize_advanced_box({"teams": {"ppa": [{"plays": 1}]}}, 1)
