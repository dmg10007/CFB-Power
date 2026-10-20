"""Results-tab worker and download verification tests; no Streamlit/browser needed."""
import hashlib
import json
import queue
from unittest.mock import Mock

import pytest
from cfb_power.results_ui import ARTIFACTS, report_worker, verified_report


def saved_report(tmp_path):
    entries = []
    for name in ARTIFACTS:
        content = b"synthetic artifact"
        (tmp_path / name).write_bytes(content)
        entries.append({"file": name, "sha256": hashlib.sha256(content).hexdigest()})
    manifest = {"kind": "weekly_results_report", "status": "ready", "schema_version": "results-report-v1", "outputs": entries}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_verification_returns_exact_download_bytes(tmp_path):
    saved_report(tmp_path)
    manifest, files = verified_report(tmp_path)
    assert manifest["status"] == "ready" and set(files) == ARTIFACTS | {"manifest.json"}
    assert files["games.csv"] == b"synthetic artifact"


def test_tampered_artifact_blocks_download(tmp_path):
    saved_report(tmp_path)
    (tmp_path / "games.csv").write_text("tampered")
    with pytest.raises(ValueError, match="blocked"):
        verified_report(tmp_path)


@pytest.mark.parametrize("field,value", [("kind", "other"), ("status", "failed"), ("schema_version", "unknown")])
def test_wrong_report_contract_blocked(tmp_path, field, value):
    manifest = saved_report(tmp_path)
    manifest[field] = value
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verified_report(tmp_path)


def test_unexpected_output_filename_is_not_followed(tmp_path):
    manifest = saved_report(tmp_path)
    manifest["outputs"][0]["file"] = "../private-file"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verified_report(tmp_path)


def test_worker_passes_scope_and_never_regenerates_forecasts(tmp_path):
    events = queue.Queue()
    client = object()
    factory = Mock(return_value=client)
    generator = Mock(return_value=tmp_path / "report")
    report_worker(tmp_path, 2026, 7, "regular", "fake-key", events, factory, generator)
    factory.assert_called_once_with(api_key="fake-key")
    generator.assert_called_once_with(data_dir=tmp_path, season=2026, week=7, season_type="regular", client=client)
    assert events.get()[0] == "log"
    assert events.get() == ("done", str(tmp_path / "report"))


def test_missing_key_does_not_construct_client(tmp_path):
    events, factory, generator = queue.Queue(), Mock(), Mock()
    report_worker(tmp_path, 2026, 7, "regular", "", events, factory, generator)
    assert events.get()[0] == "error"
    factory.assert_not_called()
    generator.assert_not_called()


def test_unknown_exception_detail_is_hidden(tmp_path):
    events = queue.Queue()
    report_worker(tmp_path, 2026, 7, "regular", "fake-key", events, Mock(), Mock(side_effect=RuntimeError("private body fake-key")))
    events.get()
    kind, message = events.get()
    assert kind == "error" and "private body" not in message and "fake-key" not in message


def test_controlled_error_redacts_key(tmp_path):
    events = queue.Queue()
    report_worker(tmp_path, 2026, 7, "regular", "fake-key", events, Mock(), Mock(side_effect=ValueError("bad fake-key")), (ValueError,))
    events.get()
    assert events.get() == ("error", "bad [REDACTED]")
