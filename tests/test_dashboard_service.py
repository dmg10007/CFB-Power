from unittest.mock import Mock

import pytest

from cfb_power.dashboard_service import read_key, refresh, save_key


def test_save_preserves_settings_and_is_private(tmp_path):
    path = tmp_path / ".env"
    path.write_text("CFB_DATA_DIR=data\nOTHER=value\n")
    save_key(tmp_path, "test-secret")
    assert read_key(tmp_path) == "test-secret"
    assert "OTHER=value" in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".env-save-*"))


def test_reject_bad_key(tmp_path):
    with pytest.raises(ValueError):
        save_key(tmp_path, "bad key")
    assert not (tmp_path / ".env").exists()


def test_reject_symlink(tmp_path):
    target = tmp_path / "target"
    target.write_text("unchanged")
    (tmp_path / ".env").symlink_to(target)
    with pytest.raises(ValueError):
        save_key(tmp_path, "secret")
    assert target.read_text() == "unchanged"


def test_capture_failure_stops_pipeline(tmp_path):
    capture, discover, assemble, forecast = (
        Mock(side_effect=RuntimeError("failed")),
        Mock(),
        Mock(),
        Mock(),
    )
    with pytest.raises(RuntimeError):
        refresh(
            data_dir=tmp_path,
            season=2026,
            weeks=[6, 7],
            key="fake",
            emit=Mock(),
            capture=capture,
            discover=discover,
            assemble=assemble,
            forecast=forecast,
        )
    discover.assert_not_called()
    assemble.assert_not_called()
    forecast.assert_not_called()


def test_pipeline_uses_returned_dataset(tmp_path):
    capture, discover, assemble, forecast = (
        Mock(),
        Mock(return_value=["snapshot"]),
        Mock(return_value="exact-dataset"),
        Mock(return_value="run"),
    )
    assert (
        refresh(
            data_dir=tmp_path,
            season=2026,
            weeks=[7, 6, 6],
            key="fake",
            emit=Mock(),
            capture=capture,
            discover=discover,
            assemble=assemble,
            forecast=forecast,
        )
        == "run"
    )
    assert [call.kwargs["week"] for call in capture.call_args_list] == [6, 7]
    forecast.assert_called_once_with("exact-dataset", tmp_path)
