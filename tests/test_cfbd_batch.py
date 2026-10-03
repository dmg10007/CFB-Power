"""Mocked retrieval and real-fixture publication tests; no API key needed."""

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from cfb_power.ingestion.cfbd_batch import BatchIngestionError, ingest_cfbd_batch, main

FIXTURES = Path(__file__).parent / "fixtures" / "cfbd"


@pytest.fixture()
def responses():
    return (
        json.loads((FIXTURES / "canonical_games_401856766.json").read_text(encoding="utf-8")),
        json.loads((FIXTURES / "canonical_team_stats_401856766.json").read_text(encoding="utf-8")),
    )


def run(tmp_path, responses, **kwargs):
    client = Mock()
    client.get_json.side_effect = list(responses)
    output = ingest_cfbd_batch(season=2026, season_type="regular", week=1, data_dir=tmp_path, client=client, **kwargs)
    return output, client


def test_archive_hashes_mapping_and_active_inputs_untouched(tmp_path, responses):
    active = tmp_path / "processed"
    active.mkdir()
    (active / "games.csv").write_text("existing games", encoding="utf-8")
    (active / "team_game_stats.csv").write_text("existing stats", encoding="utf-8")
    output, client = run(tmp_path, responses)
    assert client.get_json.call_args_list[0].args == ("/games", {"year": 2026, "seasonType": "regular", "classification": "fbs", "week": 1})
    assert client.get_json.call_args_list[1].args == ("/games/teams", {"id": 401856766})
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "ready"
    assert manifest["counts"]["team_stat_rows"] == 2
    raw = Path(manifest["raw_archive"])
    assert json.loads((raw / "games.json").read_text()) == responses[0]
    assert json.loads((raw / "box_401856766.json").read_text()) == responses[1]
    for request in manifest["requests"]:
        assert hashlib.sha256((raw / request["file"]).read_bytes()).hexdigest() == request["sha256"]
    for artifact in manifest["outputs"]:
        assert hashlib.sha256((output / artifact["file"]).read_bytes()).hexdigest() == artifact["sha256"]
    stats = pd.read_csv(output / "team_game_stats.csv").set_index("team_id")
    assert stats.loc[2628, "offensive_plays"] == 71
    assert stats.loc[153, "time_of_possession_seconds"] == 1618
    assert stats["red_zone_attempts"].isna().all()
    assert pd.read_csv(output / "games.csv")["season_type"].eq("regular").all()
    assert (active / "games.csv").read_text() == "existing games"
    assert (active / "team_game_stats.csv").read_text() == "existing stats"
    assert not list((active / "cfbd").glob(".*.staging"))


def test_exclusions_are_explicit_and_upcoming_games_have_no_box_request(tmp_path, responses):
    schedule, boxes = responses
    non_fbs, tbd, live, upcoming = [copy.deepcopy(schedule[0]) for _ in range(4)]
    non_fbs.update(id=2, awayClassification="fcs")
    tbd.update(id=3, completed=False, homePoints=None, awayPoints=None, startTimeTBD=True)
    live.update(id=4, completed=False)
    upcoming.update(id=5, completed=False, homePoints=None, awayPoints=None)
    output, client = run(tmp_path, (schedule + [non_fbs, tbd, live, upcoming], boxes))
    manifest = json.loads((output / "manifest.json").read_text())
    assert {item["reason"] for item in manifest["exclusions"]} == {"non_fbs_matchup", "unconfirmed_upcoming_kickoff", "unfinished_with_scores"}
    assert manifest["counts"]["selected_games"] == 2
    assert manifest["counts"]["upcoming_games"] == 1
    assert client.get_json.call_count == 2


@pytest.mark.parametrize("case", ["empty", "missing_class", "outside_scope", "duplicate", "invalid_box", "missing_stat", "api_error"])
def test_failure_retains_audit_and_does_not_publish(tmp_path, responses, case):
    schedule, boxes = responses
    client = Mock()
    if case == "empty":
        schedule.clear()
    elif case == "missing_class":
        schedule[0].pop("awayClassification")
    elif case == "outside_scope":
        schedule[0]["seasonType"] = "postseason"
    elif case == "duplicate":
        schedule.append(copy.deepcopy(schedule[0]))
    elif case == "invalid_box":
        boxes[0]["id"] = 999
    elif case == "missing_stat":
        boxes[0]["teams"][0]["stats"] = []
    client.get_json.side_effect = [schedule, RuntimeError("private-test-secret") if case == "api_error" else boxes]
    with pytest.raises(BatchIngestionError) as error:
        ingest_cfbd_batch(season=2026, season_type="regular", week=1, data_dir=tmp_path, client=client)
    assert "private-test-secret" not in str(error.value)
    assert not list((tmp_path / "processed" / "cfbd").glob("*/manifest.json"))
    audits = list((tmp_path / "raw" / "cfbd" / "2026").glob("*/manifest.json"))
    assert len(audits) == 1
    assert json.loads(audits[0].read_text())["status"] == "failed"
    assert "private-test-secret" not in audits[0].read_text()


def test_completed_game_cap_stops_before_box_requests(tmp_path, responses):
    schedule, _ = responses
    second = copy.deepcopy(schedule[0])
    second["id"] = 2
    client = Mock()
    client.get_json.return_value = schedule + [second]
    with pytest.raises(BatchIngestionError):
        ingest_cfbd_batch(season=2026, season_type="regular", week=1, data_dir=tmp_path, max_completed_games=1, client=client)
    assert client.get_json.call_count == 1


def test_repeated_runs_create_separate_snapshots(tmp_path, responses):
    first, _ = run(tmp_path, responses)
    second, _ = run(tmp_path, responses)
    assert first != second
    assert (first / "manifest.json").exists() and (second / "manifest.json").exists()


def test_publication_failure_removes_staging_but_keeps_raw(tmp_path, responses):
    with patch("cfb_power.ingestion.cfbd_batch.Path.rename", side_effect=OSError("publication error")):
        with pytest.raises(BatchIngestionError):
            run(tmp_path, responses)
    assert not list((tmp_path / "processed" / "cfbd").iterdir())
    audits = list((tmp_path / "raw" / "cfbd" / "2026").glob("*/manifest.json"))
    assert json.loads(audits[0].read_text())["status"] == "failed"
    assert list(audits[0].parent.glob("box_*.json"))


def test_cli_reports_ready_snapshot_without_network(tmp_path, responses, capsys):
    client = Mock()
    client.get_json.side_effect = list(responses)
    with patch("cfb_power.ingestion.cfbd_batch.CfbdClient", return_value=client):
        assert main(["--season", "2026", "--season-type", "regular", "--week", "1", "--data-dir", str(tmp_path)]) == 0
    assert "Active model input CSVs were not changed" in capsys.readouterr().out


@pytest.mark.parametrize("arguments", [{"season": True}, {"week": -1}, {"season_type": "both"}, {"max_completed_games": 0}])
def test_invalid_arguments_do_not_call_api(tmp_path, arguments):
    client = Mock()
    values = {"season": 2026, "season_type": "regular", "data_dir": tmp_path, "client": client, **arguments}
    with pytest.raises(ValueError):
        ingest_cfbd_batch(**values)
    client.get_json.assert_not_called()
