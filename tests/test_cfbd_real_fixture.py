"""Regression coverage using a trimmed user-supplied real CFBD response."""

import copy
import json
import re
from pathlib import Path

import pytest

from cfb_power.ingestion.cfbd_advanced import normalize_advanced_box

GAME_ID = 401856766
FIXTURE = Path(__file__).parent / "fixtures" / "cfbd" / "advanced_box_real_401856766_trimmed.json"


@pytest.fixture()
def real_payload():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_real_fixture_teams_counts_and_identifiers(real_payload):
    rows = normalize_advanced_box(real_payload, GAME_ID)
    assert len(rows) == 2
    assert {row["team"] for row in rows} == {"North Carolina", "TCU"}
    for row in rows:
        assert row["game_id"] == GAME_ID
        assert row["passing_game_id"] == GAME_ID
        assert row["rushing_advanced_game_id"] == GAME_ID
        assert len(set(row) - {"game_id", "team"}) == 110


def test_real_fixture_selected_values(real_payload):
    rows = {row["team"]: row for row in normalize_advanced_box(real_payload, GAME_ID)}
    unc, tcu = rows["North Carolina"], rows["TCU"]
    assert unc["ppa_plays"] == 58
    assert tcu["ppa_plays"] == 72
    assert unc["ppa_overall_total"] == pytest.approx(-0.0143)
    assert tcu["ppa_overall_total"] == pytest.approx(-0.0746)
    assert unc["success_rates_overall_total"] == pytest.approx(0.466)
    assert tcu["success_rates_overall_total"] == pytest.approx(0.333)
    assert unc["havoc_total"] == pytest.approx(0.153)
    assert tcu["havoc_total"] == pytest.approx(0.155)
    assert unc["rushing_advanced_offense_directions_left_carries"] == 7
    assert tcu["passing_offense_attempts"] == 32


def test_real_fixture_null_zero_and_legacy_spaced_keys(real_payload):
    rows = {row["team"]: row for row in normalize_advanced_box(real_payload, GAME_ID)}
    unc = rows["North Carolina"]
    assert "passing_offense_locations_unknown_total_yards" in unc
    assert unc["passing_offense_locations_unknown_total_yards"] is None
    assert unc["ppa_overall_quarter4"] == 0
    assert unc["passing_offense_locations_deep right_ppa"] == pytest.approx(-0.946)


def test_real_fixture_category_and_team_order_do_not_change_rows(real_payload):
    reordered = copy.deepcopy(real_payload)
    reordered["teams"] = {
        category: list(reversed(entries))
        for category, entries in reversed(list(reordered["teams"].items()))
    }
    original = {row["team"]: row for row in normalize_advanced_box(real_payload, GAME_ID)}
    actual = {row["team"]: row for row in normalize_advanced_box(reordered, GAME_ID)}
    assert actual == original


def test_real_fixture_preserves_every_retained_scalar(real_payload):
    expected = {team: {"game_id": GAME_ID, "team": team} for team in ("North Carolina", "TCU")}

    def visit(value, parts, output):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, parts + (key,), output)
        else:
            assert value is None or isinstance(value, (int, float, str, bool))
            column = "_".join(re.sub(r"(?<!^)(?=[A-Z])", "_", part).lower() for part in parts)
            assert column not in output, f"Colliding source paths: {column}"
            output[column] = value

    for category, entries in real_payload["teams"].items():
        for entry in entries:
            for field, value in entry.items():
                if field != "team":
                    visit(value, (category, field), expected[entry["team"]])
    actual = {row["team"]: row for row in normalize_advanced_box(real_payload, GAME_ID)}
    assert actual == expected
