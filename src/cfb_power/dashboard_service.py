"""Local dashboard settings and workflow orchestration."""
from pathlib import Path
import os
import tempfile
from dotenv import dotenv_values, set_key


def read_key(root):
    return dotenv_values(Path(root) / ".env").get("CFBD_API_KEY") or os.getenv("CFBD_API_KEY", "")


def save_key(root, key):
    key = key.strip()
    if not key or any(c.isspace() for c in key):
        raise ValueError("Enter a nonempty API key without whitespace.")
    path = Path(root) / ".env"
    if path.is_symlink():
        raise ValueError("Refusing to replace a symlinked .env.")
    fd, name = tempfile.mkstemp(prefix=".env-save-", dir=root)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            if path.exists():
                handle.write(path.read_text())
        set_key(str(temporary), "CFBD_API_KEY", key)
        os.chmod(temporary, 0o600)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def refresh(*, data_dir, season, weeks, key, emit, capture, discover, assemble, forecast):
    """Injected operations make failure ordering testable without network requests."""
    if not key:
        raise ValueError("Save an API key in Settings first.")
    from cfb_power.ingestion.cfbd_client import CfbdClient
    client = CfbdClient(api_key=key)
    for week in sorted(set(weeks)):
        emit(f"Capturing {season} regular season, week {week}")
        capture(season=season, season_type="regular", week=week,
                data_dir=data_dir, client=client)
    emit("Verifying and assembling snapshots")
    dataset = assemble(discover(data_dir), data_dir)
    emit("Generating forecasts")
    return forecast(dataset, data_dir)
