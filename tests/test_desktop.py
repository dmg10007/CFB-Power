"""Offline Linux launcher tests; no real browser, Streamlit server, or API calls."""

import os
import struct
import subprocess
from unittest.mock import MagicMock, Mock

import pytest

from cfb_power import desktop as app


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "Project With Spaces"
    python = root / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("test executable placeholder")
    python.chmod(0o700)
    return root


def test_desktop_exec_preserves_venv_and_spaces(checkout):
    text = app.desktop_text(checkout)
    assert (
        f'Exec="{checkout}/.venv/bin/python" "{checkout}/src/cfb_power/desktop.py" start --port 8501'
        in text
    )
    assert "Terminal=false" in text and "Actions=Stop;" in text
    assert "[Desktop Action Stop]" in text
    assert " stop --port 8501" in text
    assert "-m streamlit" not in text


@pytest.mark.parametrize("bad", ["a\nb", "a%b", 'a"b', "a$b", "a`b", "a\\b", "a=b"])
def test_reserved_paths_rejected(bad):
    with pytest.raises(ValueError):
        app.quote(bad)


def test_install_and_uninstall_are_scoped(checkout, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    paths = app.install(root=checkout)
    assert len(paths) == 2
    assert all(path.parent == tmp_path / "share/applications" for path in paths)
    assert all(path.stat().st_mode & 0o777 == 0o755 for path in paths)
    app.install(root=checkout)
    assert all(path.is_file() for path in paths)
    app.install(root=checkout, remove=True)
    assert not any(path.exists() for path in paths)


def test_foreign_launcher_not_overwritten(checkout, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    path = tmp_path / "share/applications/cfb-power.desktop"
    path.parent.mkdir(parents=True)
    path.write_text("unrelated user launcher")
    with pytest.raises(ValueError, match="another checkout"):
        app.install(root=checkout)
    assert path.read_text() == "unrelated user launcher"


def test_symlink_launcher_not_followed(checkout, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    path = tmp_path / "share/applications/cfb-power.desktop"
    path.parent.mkdir(parents=True)
    target = tmp_path / "target"
    target.write_text("unchanged")
    path.symlink_to(target)
    with pytest.raises(ValueError):
        app.install(root=checkout)
    assert target.read_text() == "unchanged"


def test_optional_desktop_uses_xdg_location(checkout, tmp_path, monkeypatch):
    desktop = tmp_path / "Localized Desktop"
    desktop.mkdir()
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    monkeypatch.setattr(app.shutil, "which", lambda _: "/usr/bin/xdg-user-dir")
    monkeypatch.setattr(
        app.subprocess, "run", Mock(return_value=Mock(returncode=0, stdout=str(desktop) + "\n"))
    )
    paths = app.install(root=checkout, desktop=True)
    assert paths[-1] == desktop / "cfb-power.desktop"
    assert paths[-1].is_file()


def test_streamlit_command_is_local_headless(checkout):
    command = app.streamlit_command(checkout, 8501)
    assert command[:4] == [str(checkout / ".venv/bin/python"), "-m", "streamlit", "run"]
    assert "--server.address=127.0.0.1" in command
    assert "--server.headless=true" in command
    assert "--browser.gatherUsageStats=false" in command
    assert "--server.port=8501" in command


def test_missing_venv_is_reported(tmp_path):
    with pytest.raises(ValueError, match="virtual environment"):
        app.python_path(tmp_path)


def test_private_directory_rejects_symlink_and_open_permissions(tmp_path):
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError):
        app.private_dir(link)
    target.chmod(0o755)
    with pytest.raises(ValueError):
        app.private_dir(target)


def test_repeated_start_reuses_control_socket(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    monkeypatch.setattr(app, "state_dir", lambda *args: state)
    monkeypatch.setattr(app.fcntl, "flock", Mock(side_effect=BlockingIOError))
    control = Mock()
    spawn = Mock()
    monkeypatch.setattr(app, "control", control)
    monkeypatch.setattr(app.subprocess, "Popen", spawn)
    app.start(root=tmp_path)
    control.assert_called_once_with(state, b"open")
    spawn.assert_not_called()


def test_occupied_port_does_not_spawn_or_stop_process(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    monkeypatch.setattr(app, "state_dir", lambda *args: state)
    monkeypatch.setattr(app, "port_free", lambda _: False)
    spawn, control = Mock(), Mock()
    monkeypatch.setattr(app.subprocess, "Popen", spawn)
    monkeypatch.setattr(app, "control", control)
    with pytest.raises(RuntimeError, match="occupied"):
        app.start(root=tmp_path)
    spawn.assert_not_called()
    control.assert_not_called()


def connection(command, uid=None):
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.getsockopt.return_value = struct.pack(
        "3i", 123, os.getuid() if uid is None else uid, os.getgid()
    )
    conn.recv.return_value = command
    return conn


def test_supervisor_open_then_stop_controls_owned_child(monkeypatch):
    child, listener = Mock(), Mock()
    child.poll.return_value = None
    open_conn, stop_conn = connection(b"open"), connection(b"stop")
    listener.accept.side_effect = [(open_conn, None), (stop_conn, None)]
    monkeypatch.setattr(app, "healthy", lambda _: True)
    browser = Mock(return_value=True)
    monkeypatch.setattr(app.webbrowser, "open", browser)
    app.serve(child, listener, 8501)
    assert browser.call_count == 2
    stop_conn.sendall.assert_called_once_with(b"ok")
    child.terminate.assert_called_once()
    child.wait.assert_called_once_with(timeout=10)


def test_supervisor_rejects_foreign_uid(monkeypatch):
    child, listener = Mock(), Mock()
    child.poll.return_value = None
    foreign, own = connection(b"stop", os.getuid() + 1), connection(b"stop")
    listener.accept.side_effect = [(foreign, None), (own, None)]
    monkeypatch.setattr(app, "healthy", lambda _: True)
    monkeypatch.setattr(app.webbrowser, "open", Mock(return_value=True))
    app.serve(child, listener, 8501)
    foreign.recv.assert_not_called()
    own.recv.assert_called_once()


def test_startup_timeout_stops_child(monkeypatch):
    child = Mock()
    child.poll.return_value = None
    monkeypatch.setattr(app.time, "monotonic", Mock(side_effect=[0, 46]))
    monkeypatch.setattr(app, "healthy", lambda _: False)
    with pytest.raises(RuntimeError, match="timed out"):
        app.serve(child, Mock(), 8501)
    child.terminate.assert_called_once()


def test_exit_is_reported_without_killing_other_processes():
    child = Mock()
    child.poll.return_value = 1
    with pytest.raises(RuntimeError, match="exited"):
        app.serve(child, Mock(), 8501)
    child.terminate.assert_not_called()


def test_shutdown_escalates_only_for_owned_child():
    child = Mock()
    child.poll.return_value = None
    child.wait.side_effect = [subprocess.TimeoutExpired("streamlit", 10), None]
    app.stop_child(child)
    child.kill.assert_called_once()


@pytest.mark.parametrize(
    "status, body, expected", [(200, b"ok", True), (200, b"other", False), (404, b"ok", False)]
)
def test_health_requires_streamlit_response(monkeypatch, status, body, expected):
    response = MagicMock()
    response.__enter__.return_value = response
    response.status = status
    response.read.return_value = body
    opener = Mock()
    opener.open.return_value = response
    monkeypatch.setattr(app.urllib.request, "build_opener", Mock(return_value=opener))
    assert app.healthy(8501) is expected


def test_stop_without_managed_server_does_not_signal(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(app, "state_dir", lambda *args: tmp_path)
    monkeypatch.setattr(app, "control", Mock(side_effect=FileNotFoundError))
    assert app.main(["stop"]) == 0
    assert "Manual servers are not stopped" in capsys.readouterr().out


def test_port_probe_allows_clean_restart(monkeypatch):
    server = MagicMock()
    server.__enter__.return_value = server
    monkeypatch.setattr(app.socket, "socket", Mock(return_value=server))
    assert app.port_free(8501) is True
    server.setsockopt.assert_called_once_with(app.socket.SOL_SOCKET, app.socket.SO_REUSEADDR, 1)
    server.bind.assert_called_once_with(("127.0.0.1", 8501))
