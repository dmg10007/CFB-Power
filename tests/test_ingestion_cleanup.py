"""Offline HTTP diagnostics, probe validation, and ignore-rule tests."""

import io
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest

from cfb_power import probe
from cfb_power.ingestion import cfbd_client as api
from cfb_power.ingestion.cfbd_batch import BatchIngestionError, ingest_cfbd_batch


@pytest.mark.parametrize("status", [401, 403, 429, 503])
def test_http_error_excludes_private_fields(monkeypatch, status):
    error = HTTPError(
        "https://example.invalid/?key=fake-private-key",
        status,
        "fake-private-key",
        {"Authorization": "Bearer fake-private-key"},
        io.BytesIO(b"private response body fake-private-key"),
    )
    monkeypatch.setattr(api.urllib.request, "urlopen", Mock(side_effect=error))
    monkeypatch.setattr(api.time, "sleep", Mock())
    with pytest.raises(api.CfbdHttpError) as caught:
        api.CfbdClient(api_key="fake-private-key").get_json("/games", {"year": 2026})
    assert str(caught.value) == f"CFBD request failed: HTTP {status}; endpoint /games"
    assert caught.value.status_code == status
    assert caught.value.__suppress_context__ is True
    assert "fake-private-key" not in str(caught.value)
    assert "private response body" not in str(caught.value)


@pytest.mark.parametrize(
    "endpoint", ["/games?key=fake-private-key", "https://example.invalid/fake-private-key"]
)
def test_unknown_endpoints_are_redacted(endpoint):
    error = api.CfbdHttpError(403, endpoint)
    assert error.endpoint == "[redacted endpoint]"
    assert "fake-private-key" not in str(error)


def test_batch_preserves_safe_http_status_and_audit(tmp_path):
    client = Mock()
    client.get_json.side_effect = api.CfbdHttpError(429, "/games")
    with pytest.raises(BatchIngestionError) as caught:
        ingest_cfbd_batch(season=2026, season_type="regular", data_dir=tmp_path, client=client)
    audit = next((tmp_path / "raw/cfbd/2026").glob("*/manifest.json"))
    manifest = json.loads(audit.read_text())
    assert manifest["status"] == "failed"
    assert manifest["http_error"] == {"status_code": 429, "endpoint": "/games"}
    assert "HTTP 429" in str(caught.value) and str(audit.parent) in str(caught.value)
    assert not (tmp_path / "processed").exists()


def test_generic_prefix_cannot_expose_private_error(tmp_path):
    client = Mock()
    client.get_json.side_effect = RuntimeError("CFBD request failed: HTTP fake-private-key")
    with pytest.raises(BatchIngestionError) as caught:
        ingest_cfbd_batch(season=2026, season_type="regular", data_dir=tmp_path, client=client)
    assert "fake-private-key" not in str(caught.value)
    audit = next((tmp_path / "raw/cfbd/2026").glob("*/manifest.json"))
    assert "fake-private-key" not in audit.read_text()
    assert "http_error" not in json.loads(audit.read_text())


def payloads():
    games = [
        {
            "id": 123,
            "completed": True,
            "season": 2026,
            "seasonType": "regular",
            "homeClassification": "fbs",
            "awayClassification": "fbs",
            "homeId": 1,
            "awayId": 2,
        }
    ]
    boxes = [
        {
            "id": 123,
            "teams": [
                {"teamId": 1, "stats": [{"category": "thirdDownEff", "stat": "2-8"}]},
                {"teamId": 2, "stats": [{"category": "thirdDownEff", "stat": "3-9"}]},
            ],
        }
    ]
    return games, boxes, {"teams": {}}


def test_probe_unique_paths_id_only_nested_stats(tmp_path, monkeypatch, capsys):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    outputs = []
    for _ in range(2):
        client = Mock()
        client.get_json.side_effect = payloads()
        outputs.append(probe.probe(season=2026, root=tmp_path, client=client))
        assert client.get_json.call_args_list[1].args == ("/games/teams", {"id": 123})
        assert client.get_json.call_args_list[2].args == ("/game/box/advanced", {"id": 123})
    assert outputs[0] != outputs[1]
    assert all(p.parent == tmp_path / "data/probes/cfbd" for p in outputs)
    assert all((p / "team_stats_123.json").exists() for p in outputs)
    assert "Team 1: 1 stats; fields: category, stat" in capsys.readouterr().out
    assert not (elsewhere / "data").exists()


@pytest.mark.parametrize("case", ["wrong_id", "multiple", "wrong_teams", "bad_stats"])
def test_probe_rejects_invalid_box_before_advanced(tmp_path, case):
    games, boxes, _ = payloads()
    if case == "wrong_id":
        boxes[0]["id"] = 999
    elif case == "multiple":
        boxes.append(boxes[0])
    elif case == "wrong_teams":
        boxes[0]["teams"][0]["teamId"] = 999
    else:
        boxes[0]["teams"][0]["stats"] = {}
    client = Mock()
    client.get_json.side_effect = [games, boxes]
    with pytest.raises(ValueError):
        probe.probe(season=2026, root=tmp_path, client=client)
    assert client.get_json.call_count == 2
    assert not list((tmp_path / "data/probes/cfbd").glob("*/team_stats_*.json"))


@pytest.mark.parametrize(
    "args",
    [{"season": True}, {"season": 2026, "game_id": -1}, {"season": 2026, "season_type": "both"}],
)
def test_probe_invalid_arguments_make_no_requests(tmp_path, args):
    client = Mock()
    with pytest.raises(ValueError):
        probe.probe(root=tmp_path, client=client, **args)
    client.get_json.assert_not_called()


def test_probe_missing_key(monkeypatch, tmp_path):
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    with pytest.raises(ValueError, match="missing"):
        probe.probe(season=2026, root=tmp_path)
    assert not (tmp_path / "data").exists()


def test_probe_cli_does_not_echo_unknown_exception(monkeypatch, capsys):
    monkeypatch.setattr(probe, "probe", Mock(side_effect=RuntimeError("fake-private-key")))
    assert probe.main(["--season", "2026"]) == 1
    output = capsys.readouterr()
    assert "fake-private-key" not in output.out + output.err
    assert "RuntimeError" in output.err


def test_probe_reads_project_env(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("CFBD_API_KEY=local-fake-key\n")
    monkeypatch.setenv("CFBD_API_KEY", "different-env-key")
    client = Mock()
    client.get_json.side_effect = payloads()
    constructor = Mock(return_value=client)
    monkeypatch.setattr(probe, "CfbdClient", constructor)
    probe.probe(season=2026, root=tmp_path)
    constructor.assert_called_once_with(api_key="local-fake-key")


def test_ignore_outputs_not_source(tmp_path):
    rules = Path(__file__).resolve().parents[1] / ".gitignore"
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text(rules.read_text())
    ignored = [
        "data/probes/cfbd/payload.json",
        "data/datasets/a/games.csv",
        "data/runs/a/predictions.csv",
        "src/cfb_power.egg-info/PKG-INFO",
        ".env",
    ]
    for path in ignored:
        assert (
            subprocess.run(["git", "-C", str(tmp_path), "check-ignore", "-q", path]).returncode == 0
        )
    assert (
        subprocess.run(
            ["git", "-C", str(tmp_path), "check-ignore", "-q", "scripts/probe_cfbd.py"]
        ).returncode
        == 1
    )
