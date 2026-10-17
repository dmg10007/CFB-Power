"""Linux-only, per-user launcher for an editable CFB Power checkout."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[2]


def quote(value):
    text = str(value)
    if any(char in text for char in '\n\r\t%\\"`$='):
        raise ValueError("Launcher paths cannot contain control characters or % \\ \" ` $ =.")
    return '"' + text + '"'


def python_path(root):
    path = Path(root) / ".venv/bin/python"
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("Project .venv/bin/python is missing. Create/install the virtual environment first.")
    return path.absolute()


def desktop_text(root, port=8501, stop=False):
    root = Path(root).resolve()
    command = f"{quote(python_path(root))} {quote(root / 'src/cfb_power/desktop.py')}"
    title = "CFB Power - Stop" if stop else "CFB Power"
    action = "stop" if stop else "start"
    text = (f"[Desktop Entry]\nType=Application\nName={title}\n"
            "Comment=Local college football forecasts\nIcon=applications-science\n"
            f"Exec={command} {action} --port {port}\nTerminal=false\nStartupNotify=false\n"
            f"Categories=Education;Science;\nX-CFB-Power-Root={root}\n")
    if not stop:
        text += ("Actions=Stop;\n\n[Desktop Action Stop]\nName=Stop dashboard\n"
                 f"Exec={command} stop --port {port}\n")
    return text


def locations(desktop=False):
    home = Path.home()
    base = Path(os.environ.get("XDG_DATA_HOME", str(home / ".local/share")))
    if not base.is_absolute():
        base = home / ".local/share"
    paths = [base / "applications/cfb-power.desktop", base / "applications/cfb-power-stop.desktop"]
    if desktop:
        executable = shutil.which("xdg-user-dir")
        folder = home / "Desktop"
        if executable:
            result = subprocess.run([executable, "DESKTOP"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                folder = Path(result.stdout.strip())
        if not folder.is_absolute() or not folder.is_dir():
            raise ValueError("Desktop folder not found; install without --desktop for menu entries only.")
        paths.append(folder / "cfb-power.desktop")
    return paths


def install(root=ROOT, port=8501, desktop=False, remove=False):
    root = Path(root).resolve()
    quote(root)
    paths = locations(desktop)
    marker = f"X-CFB-Power-Root={root}"
    for path in paths:
        if path.is_symlink() or (path.exists() and marker not in path.read_text().splitlines()):
            raise ValueError("Refusing to replace/remove a launcher belonging to another checkout.")
    if remove:
        for path in paths:
            path.unlink(missing_ok=True)
        return paths
    contents = [desktop_text(root, port, path.name == "cfb-power-stop.desktop") for path in paths]
    for path, content in zip(paths, contents):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".cfb-launcher-", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o755)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return paths


def private_dir(path):
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Launcher state directory must be owned by you and private (mode 700).")
    return path


def state_dir(root, port):
    identity = hashlib.sha256(f"{Path(root).resolve()}:{port}".encode()).hexdigest()[:16]
    return private_dir(Path(tempfile.gettempdir()) / f"cfb-power-{os.getuid()}-{identity}")


def control(directory, command):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(3)
        client.connect(str(directory / "control.sock"))
        client.sendall(command)
        if client.recv(16) != b"ok":
            raise RuntimeError("Launcher did not accept the request.")


def port_free(port):
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def healthy(port):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{port}/_stcore/health", timeout=0.5) as response:
            return response.status == 200 and response.read(16).strip() == b"ok"
    except OSError:
        return False


def streamlit_command(root, port):
    return [str(python_path(root)), "-m", "streamlit", "run", str(Path(root) / "src/cfb_power/dashboard.py"),
            "--server.address=127.0.0.1", f"--server.port={port}", "--server.headless=true",
            "--browser.gatherUsageStats=false"]


def stop_child(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def serve(child, listener, port, timeout=45):
    deadline = time.monotonic() + timeout
    ready, open_pending = False, True
    listener.settimeout(0.2)
    try:
        while child.poll() is None:
            if not ready:
                ready = healthy(port)
                if not ready and time.monotonic() >= deadline:
                    raise RuntimeError("Dashboard startup timed out; inspect the private launcher log.")
            if ready and open_pending:
                if not webbrowser.open(f"http://127.0.0.1:{port}"):
                    notify(f"Dashboard ready at http://127.0.0.1:{port}; open this URL in your browser.")
                open_pending = False
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            with connection:
                connection.settimeout(1)
                credentials = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
                if struct.unpack("3i", credentials)[1] != os.getuid():
                    continue
                try:
                    request = connection.recv(16)
                    if request == b"stop":
                        connection.sendall(b"ok")
                        return
                    if request == b"open":
                        open_pending = True
                        connection.sendall(b"ok")
                except OSError:
                    continue
        raise RuntimeError("Dashboard exited; inspect the private launcher log.")
    finally:
        stop_child(child)


def start(root=ROOT, port=8501):
    root = Path(root).resolve()
    directory = state_dir(root, port)
    fd = os.open(directory / "server.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            for _ in range(20):
                try:
                    control(directory, b"open")
                    return
                except (FileNotFoundError, ConnectionRefusedError):
                    time.sleep(0.1)
            raise RuntimeError("Launcher is busy starting; try again shortly.")
        if not port_free(port):
            raise RuntimeError(f"Port {port} is occupied. Stop the manually launched server first; it was not modified.")
        command = streamlit_command(root, port)
        sockpath = directory / "control.sock"
        sockpath.unlink(missing_ok=True)
        log_base = Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state")))
        if not log_base.is_absolute():
            log_base = Path.home() / ".local/state"
        logs = private_dir(log_base / "cfb-power")
        logfile = logs / (directory.name + ".log")
        logfd = os.open(logfile, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        os.fchmod(logfd, 0o600)
        print(f"Private server log: {logfile}")
        with os.fdopen(logfd, "w") as log, socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(sockpath))
            os.chmod(sockpath, 0o600)
            listener.listen(8)
            try:
                child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                serve(child, listener, port)
            finally:
                sockpath.unlink(missing_ok=True)


def notify(message):
    executable = shutil.which("notify-send")
    if executable:
        try:
            subprocess.run([executable, "CFB Power", message], timeout=5, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass
    print(message, file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Install/start/stop the Linux CFB Power desktop launcher")
    parser.add_argument("action", choices=("install", "uninstall", "start", "stop"))
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--desktop", action="store_true", help="also install/remove a Desktop shortcut")
    args = parser.parse_args(argv)
    if not sys.platform.startswith("linux") or not 1024 <= args.port <= 65535:
        parser.error("Linux and a port from 1024 to 65535 are required.")
    try:
        if args.action in ("install", "uninstall"):
            for path in install(port=args.port, desktop=args.desktop, remove=args.action == "uninstall"):
                print(path)
        elif args.action == "start":
            start(port=args.port)
        else:
            try:
                control(state_dir(ROOT, args.port), b"stop")
                print("Stop requested for the launcher-managed dashboard.")
            except (FileNotFoundError, ConnectionRefusedError):
                print("No launcher-managed dashboard is running. Manual servers are not stopped.")
    except Exception as exc:
        notify(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
