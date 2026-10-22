"""Results Report tab; workers never access Streamlit session state."""

import hashlib
import io
import json
import queue
import threading
import time
from pathlib import Path

ARTIFACTS = frozenset(
    {
        "results.json",
        "games.csv",
        "not_scored.csv",
        "summary.csv",
        "excluded_predictions.csv",
        "results_report.xlsx",
    }
)


def verified_report(directory):
    """Read a consistent, hash-verified set of bytes for display/download."""
    directory = Path(directory)
    try:
        manifest_bytes = (directory / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        if (
            manifest["kind"] != "weekly_results_report"
            or manifest["status"] != "ready"
            or manifest["schema_version"] != "results-report-v1"
        ):
            raise ValueError
        entries = manifest["outputs"]
        if len(entries) != len(ARTIFACTS) or {item["file"] for item in entries} != ARTIFACTS:
            raise ValueError
        files = {}
        for item in entries:
            content = (directory / item["file"]).read_bytes()
            if hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise ValueError
            files[item["file"]] = content
        files["manifest.json"] = manifest_bytes
        return manifest, files
    except (OSError, ValueError, KeyError, TypeError):
        raise ValueError(
            "Report files or manifest failed verification; display and downloads blocked."
        ) from None


def report_worker(
    data_dir, season, week, season_type, key, events, client_factory, generator, safe_errors=()
):
    if not key:
        events.put(("error", "Save an API key in Settings before refreshing final scores."))
        return
    try:
        events.put(
            (
                "log",
                "Verifying saved forecasts, refreshing final scores, and building the report...",
            )
        )
        client = client_factory(api_key=key)
        output = generator(
            data_dir=data_dir, season=season, week=week, season_type=season_type, client=client
        )
        events.put(("done", str(output)))
    except Exception as exc:
        message = (
            str(exc)
            if isinstance(exc, safe_errors)
            else f"Report failed ({type(exc).__name__}). Check configuration and local file permissions."
        )
        events.put(("error", message.replace(key, "[REDACTED]")))


def render_results(root, data_dir, busy):
    import pandas as pd
    import streamlit as st

    from .dashboard_service import read_key
    from .ingestion.cfbd_client import CfbdClient, CfbdHttpError
    from .results_report import ReportError, generate_report

    st.subheader("Predictions vs results")
    st.write(
        "Automatically selects the latest eligible saved pregame prediction for each game. Predictions are never regenerated for this report."
    )
    season = int(
        st.number_input(
            "Report season",
            min_value=2000,
            max_value=2100,
            value=2026,
            key="report_season",
            disabled=busy,
        )
    )
    week = int(
        st.number_input(
            "Report week", min_value=0, max_value=30, value=7, key="report_week", disabled=busy
        )
    )
    season_type = st.selectbox(
        "Report season type", ["regular", "postseason"], key="report_season_type", disabled=busy
    )
    st.caption(
        "One schedule/final-score request per click; no box-score capture or model training. Existing reports can be viewed without a request."
    )
    if st.button("Refresh final scores & generate report", disabled=busy):
        events = queue.Queue()
        thread = threading.Thread(
            target=report_worker,
            daemon=True,
            args=(
                data_dir,
                season,
                week,
                season_type,
                read_key(root),
                events,
                CfbdClient,
                generate_report,
                (ReportError, CfbdHttpError),
            ),
        )
        st.session_state.report_job = (thread, events)
        st.session_state.report_messages = []
        thread.start()
        st.rerun()
    if "report_job" in st.session_state:
        thread, events = st.session_state.report_job
        while True:
            try:
                kind, message = events.get_nowait()
                st.session_state.report_messages.append((kind, message))
                if kind == "done":
                    st.session_state.results_history = message
            except queue.Empty:
                break
        for kind, message in st.session_state.report_messages:
            if kind == "error":
                st.error(message)
            elif kind == "done":
                st.success(f"Report published: {message}")
            else:
                st.write(message)
        if thread.is_alive():
            st.caption("Working. Keep the dashboard running until the report is published.")
            time.sleep(1)
            st.rerun()
    paths = [
        str(path)
        for path in sorted((Path(data_dir) / "reports").glob("*"), reverse=True)
        if path.is_dir() and not path.name.startswith(".") and (path / "manifest.json").is_file()
    ]
    if not paths:
        st.info("No published results reports yet. Generate one above.")
        return
    if st.session_state.get("results_history") not in paths:
        st.session_state.results_history = paths[0]
    selected = st.selectbox(
        "Saved results report",
        paths,
        format_func=lambda value: Path(value).name,
        key="results_history",
    )
    try:
        manifest, files = verified_report(selected)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.write(
        {"scope": manifest["scope"], "results_captured_at_utc": manifest["results_captured_at_utc"]}
    )
    counts = manifest["counts"]
    cards = st.columns(4)
    for card, name, field in zip(
        cards,
        ["Games in scope", "Scored", "Not scored", "Final without forecast"],
        ["games_in_scope", "scored", "not_scored", "completed_without_prediction"],
        strict=True,
    ):
        card.metric(name, counts[field])
    if manifest["partial"]:
        st.warning(
            "Partial coverage: some games are unfinished, lack an eligible pregame forecast, or have an unconfirmed kickoff. See Not Scored."
        )
    summary = pd.read_csv(io.BytesIO(files["summary.csv"]))
    games = pd.read_csv(io.BytesIO(files["games.csv"]))
    overall = summary[summary["group"] == "all"].iloc[0]
    cards = st.columns(4)
    for card, label, field, fmt in zip(
        cards,
        ["Winner accuracy", "Margin MAE", "Total MAE", "Brier score"],
        ["winner_accuracy", "margin_mae", "total_mae", "brier_score"],
        [".1%", ".1f", ".1f", ".3f"],
        strict=True,
    ):
        value = overall[field]
        card.metric(label, "—" if pd.isna(value) else format(value, fmt))
    st.caption(
        "MAE is average absolute error (lower is better). Margin is home minus away, not a sportsbook handicap. Winner accuracy excludes no-picks and final ties; Brier excludes final ties. Denominators are shown below."
    )
    st.dataframe(summary, width="stretch")
    st.subheader("Game-by-game comparison")
    if (
        "matchup" not in games.columns
        and "away_team" in games.columns
        and "home_team" in games.columns
    ):
        games["matchup"] = games["away_team"].astype(str) + " @ " + games["home_team"].astype(str)

    results_config = {
        "matchup": st.column_config.TextColumn("Matchup", width="medium"),
        "status": st.column_config.TextColumn("Status", width="small"),
        "winner_correct": st.column_config.CheckboxColumn("Pick Correct?"),
        "actual_home_points": st.column_config.NumberColumn("Actual Home", format="%d"),
        "actual_away_points": st.column_config.NumberColumn("Actual Away", format="%d"),
        "home_expected_points": st.column_config.NumberColumn("Proj Home", format="%.1f"),
        "away_expected_points": st.column_config.NumberColumn("Proj Away", format="%.1f"),
        "projected_spread_home": st.column_config.NumberColumn("Proj Spread", format="%+.1f"),
        "actual_margin_home": st.column_config.NumberColumn("Actual Margin", format="%+.0f"),
        "home_win_probability": st.column_config.ProgressColumn(
            "Home Win %", format="%.1f%%", min_value=0.0, max_value=1.0
        ),
        "margin_absolute_error": st.column_config.NumberColumn("Margin Error", format="%.1f"),
        "total_absolute_error": st.column_config.NumberColumn("Total Error", format="%.1f"),
    }
    game_cols = [c for c in results_config.keys() if c in games.columns]
    other_cols = [c for c in games.columns if c not in game_cols]
    st.dataframe(
        games[game_cols + other_cols],
        column_config=results_config,
        width="stretch",
        hide_index=True,
    )
    with st.expander("Not scored / coverage gaps"):
        st.dataframe(pd.read_csv(io.BytesIO(files["not_scored.csv"])), width="stretch")
    with st.expander("Excluded prediction candidates"):
        st.caption(
            "These candidates failed eligibility checks; a different eligible version may still have been selected for the same game."
        )
        st.dataframe(pd.read_csv(io.BytesIO(files["excluded_predictions.csv"])), width="stretch")
    with st.expander("Source forecast warnings and provenance"):
        used = set(games["source_run"].dropna())
        for item in manifest["forecast_inputs"]:
            if item["source_run"] in used:
                for warning in item.get("forecast_warnings", []):
                    st.warning(f"{Path(item['source_run']).name}: {warning}")
        st.json(manifest)
    for name in (
        "results_report.xlsx",
        "games.csv",
        "summary.csv",
        "not_scored.csv",
        "excluded_predictions.csv",
        "manifest.json",
    ):
        st.download_button(
            f"Download {name}", files[name], file_name=name, key=f"report-download-{name}"
        )
