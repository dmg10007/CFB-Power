# Linux Mint desktop launcher

This launcher is for the existing editable Linux checkout with `.venv/bin/python`. It is not a packaged standalone executable. No sudo, autostart, system service, or new Python dependency is needed; install the existing GUI requirements first.

## Install once

Stop any manually started Streamlit server with Ctrl+C first. From the project root:

```bash
.venv/bin/python -m pip install -r requirements-gui.txt
.venv/bin/python -m cfb_power.desktop install --desktop
```

Without `--desktop`, install application-menu entries only. With it, the installer uses `xdg-user-dir DESKTOP` (or an existing `~/Desktop` fallback) for the optional shortcut. Linux Mint may require right-click > Allow Launching/Mark as Trusted before the desktop shortcut runs. Search the application menu for CFB Power and CFB Power - Stop.

## Use

Click CFB Power. It uses the project's virtual-environment Python to start Streamlit on 127.0.0.1:8501 with headless mode and usage telemetry disabled. The browser opens after the health check succeeds. Repeated clicks ask the existing launcher supervisor to open the dashboard again; they do not spawn another Streamlit server. A manually started server occupying the port is not reused or terminated: stop it yourself first.

Closing the browser tab does not stop the server. Use CFB Power - Stop in the application menu, or the launcher's Stop dashboard action where supported. Do not stop during capture, assembly, or forecasting: active work may be interrupted and a partial audit/staging directory may remain. The launcher is a single-server convenience, not a cross-browser job lock.

## Troubleshooting

Desktop notifications use notify-send when available; there is no new notification dependency. If a click appears to do nothing, run:

```bash
.venv/bin/python -m cfb_power.desktop start
```

The command prints the private server log path under `$XDG_STATE_HOME/cfb-power` (default `~/.local/state/cfb-power`). Logs are owner-only and replaced on each new managed start. Treat them as private diagnostic data; do not share them without checking for secrets. This command remains running as the supervisor until Stop is requested; leave that process alive. There are no API calls in the launcher itself.

Control sockets and a lock file live in a private per-user, per-checkout/port temporary directory. Stop uses that socket and peer-UID checks; it does not kill processes by name or signal an arbitrary stored PID. Startup timeout or failure stops only the Streamlit child started by that supervisor. Abruptly killing the supervisor or logging out is outside the normal Stop flow; a leftover server may require manual inspection.

Choose a different port at install time if required, e.g. `install --desktop --port 8502`. Use that same port for manual start/stop. There is no automatic port hopping. Paths with spaces are supported; control characters and `%`, backslash, double quote, backtick, dollar sign, or equals sign are deliberately rejected instead of unsafe desktop-entry escaping.

Installation writes absolute paths and refuses to overwrite a launcher for another checkout. Uninstall before moving the repository or changing checkouts:

```bash
.venv/bin/python -m cfb_power.desktop stop
.venv/bin/python -m cfb_power.desktop uninstall --desktop
```

Uninstall removes only matching launcher entries. It does not delete .env, virtual environments, datasets, forecasts, logs, or source files, and it does not stop a server by itself. Keep the repository available while the launcher runs and avoid switching branches during active jobs.

## Validation

27 offline launcher tests passed in an isolated Linux sandbox with standard-library dependencies; browser/server/control interactions were mocked. No live Streamlit instance or native desktop menu was exercised, and desktop-file-validate was unavailable. Run the full project suite and smoke-test menu/desktop launch, a repeated click, Stop, and relaunch on Linux Mint before merging. Expected suite count is 181 (154 existing + 27), if otherwise unchanged.
