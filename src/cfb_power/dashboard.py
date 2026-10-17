"""Run with python -m streamlit run src/cfb_power/dashboard.py on localhost."""
from pathlib import Path
import json
import os
import queue
import threading
import time


def main():
    import streamlit as st
    import pandas as pd
    from dotenv import dotenv_values
    from cfb_power.dashboard_service import read_key, save_key, refresh
    from cfb_power.dataset_workflow import discover_snapshots
    from cfb_power.ingestion.cfbd_batch import ingest_cfbd_batch
    from cfb_power.ingestion.cfbd_client import CfbdClient
    from cfb_power.upcoming import predict_upcoming
    from cfb_power.revisions import ReviewRequired, assemble_or_review
    from cfb_power.revisions_ui import render_review

    root = Path(__file__).resolve().parents[2]
    settings = dotenv_values(root / ".env")
    data_dir = Path(settings.get("CFB_DATA_DIR") or os.getenv("CFB_DATA_DIR", "data"))
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    st.set_page_config(page_title="CFB Power", layout="wide")
    st.title("CFB Power")
    st.caption("Local dashboard · regular season · rolling-average forecasts")
    st.caption(f"Data folder: {data_dir}")
    settings_tab, update_tab, forecasts_tab = st.tabs(["Settings", "Update & forecast", "Forecasts & history"])
    busy = "job" in st.session_state and st.session_state.job[0].is_alive()
    with settings_tab:
        st.warning("The key is stored as plaintext in your local .env, owner-readable only. Never share that file.")
        st.write("API key saved" if read_key(root) else "No API key configured")
        with st.form("credentials", clear_on_submit=True):
            key = st.text_input("CFBD API key", type="password")
            save = st.form_submit_button("Save key", disabled=busy)
        if save:
            try:
                save_key(root, key)
                st.success("API key saved. Future dashboard requests use this key.")
            except Exception:
                st.error("Could not save key. Check input and project-folder permissions.")
        if st.button("Test saved key", disabled=busy):
            try:
                saved = read_key(root)
                if not saved:
                    raise ValueError("Missing key")
                CfbdClient(api_key=saved).get_json("/teams", {"classification": "fbs"})
                st.success("API request succeeded.")
            except Exception:
                st.error("Key test failed. Check the saved key, connection, and CFBD account quota.")
    with update_tab:
        st.write("Choose every week to refresh, including weeks with missing results. All captures must succeed before assembly starts.")
        season = int(st.number_input("Season", min_value=2000, max_value=2100, value=2026, disabled=busy))
        weeks = st.multiselect("Weeks to capture", list(range(0, 21)), default=[6, 7], disabled=busy)
        st.info("Each week uses one schedule request plus one box-score request per completed game (up to 100). Repeated runs repeat requests. Partial successful snapshots remain available if a later step fails.")
        if st.button("Update & forecast", disabled=busy or not weeks):
            events = queue.Queue()
            secret = read_key(root)
            def worker():
                try:
                    run = refresh(data_dir=data_dir, season=season, weeks=weeks, key=secret,
                                  emit=lambda message: events.put(("log", message)),
                                  capture=ingest_cfbd_batch, discover=discover_snapshots,
                                  assemble=assemble_or_review, forecast=predict_upcoming)
                    events.put(("done", str(run)))
                except ReviewRequired as exc:
                    events.put(("review", exc.plan))
                except Exception as exc:
                    message = str(exc).replace(secret, "[REDACTED]") if secret else str(exc)
                    events.put(("error", message))
            thread = threading.Thread(target=worker, daemon=True)
            st.session_state.job = (thread, events)
            st.session_state.messages = []
            thread.start()
            st.rerun()
        if "job" in st.session_state:
            thread, events = st.session_state.job
            while True:
                try:
                    kind, message = events.get_nowait()
                    if kind == "review":
                        st.session_state.review_plan = message
                        st.session_state.messages.append(("log", "Capture complete. Review revisions below before assembly."))
                    else:
                        st.session_state.messages.append((kind, message))
                except queue.Empty:
                    break
            for kind, message in st.session_state.messages:
                if kind == "error":
                    st.error(message)
                elif kind == "done":
                    st.success(f"Forecast published: {message}")
                else:
                    st.write(message)
            if thread.is_alive():
                st.caption("Working. Keep this server running; closing the app process interrupts the job.")
                time.sleep(1)
                st.rerun()
        render_review(data_dir, busy)
    with forecasts_tab:
        runs = []
        for path in sorted((data_dir / "runs").glob("*/predictions_*/manifest.json"), reverse=True):
            try:
                manifest = json.loads(path.read_text())
                if manifest.get("status") == "ready":
                    runs.append((path.parent, manifest))
            except (OSError, ValueError):
                continue
        if not runs:
            st.info("No published forecast runs yet.")
        else:
            selected = st.selectbox("Forecast run", range(len(runs)), format_func=lambda i: str(runs[i][0].relative_to(data_dir)))
            run, manifest = runs[selected]
            st.write({k: manifest.get(k) for k in ("created_at_utc", "as_of_utc", "data_through_utc", "counts")})
            for warning in manifest.get("warnings", []):
                st.warning(warning)
            from hashlib import sha256
            valid = all((run / item["file"]).is_file() and sha256((run / item["file"]).read_bytes()).hexdigest() == item["sha256"] for item in manifest["outputs"])
            if not valid:
                st.error("Run artifacts failed integrity verification; not displaying them.")
            else:
                st.caption("projected_spread_home = home points minus away points; positive favors home, unlike sportsbook handicap notation.")
                st.dataframe(pd.read_csv(run / "predictions.csv"), width="stretch")
                with st.expander("Skipped games"):
                    st.dataframe(pd.read_csv(run / "skipped.csv"), width="stretch")
                for name in ("predictions.csv", "skipped.csv", "upcoming_predictions.xlsx"):
                    st.download_button(f"Download {name}", (run / name).read_bytes(), file_name=name)


if __name__ == "__main__":
    main()
