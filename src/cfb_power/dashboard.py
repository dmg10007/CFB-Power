"""Run with python -m streamlit run src/cfb_power/dashboard.py on localhost."""

import json
import os
import queue
import threading
import time
from pathlib import Path


def main():
    import pandas as pd
    import streamlit as st
    from dotenv import dotenv_values

    from cfb_power.dashboard_power import (
        get_live_power_rankings,
        render_matchup_analyzer_tab,
        render_power_rankings_tab,
    )
    from cfb_power.dashboard_service import read_key, refresh, save_key
    from cfb_power.dataset_workflow import discover_snapshots
    from cfb_power.ingestion.cfbd_batch import ingest_cfbd_batch
    from cfb_power.ingestion.cfbd_client import CfbdClient
    from cfb_power.results_ui import render_results
    from cfb_power.revisions import ReviewRequired, assemble_or_review
    from cfb_power.revisions_ui import render_review
    from cfb_power.team_names import enrich_predictions_frame, resolve_team_names
    from cfb_power.upcoming import predict_upcoming

    root = Path(__file__).resolve().parents[2]
    settings = dotenv_values(root / ".env")
    data_dir = Path(settings.get("CFB_DATA_DIR") or os.getenv("CFB_DATA_DIR", "data"))
    if not data_dir.is_absolute():
        data_dir = root / data_dir

    st.set_page_config(
        page_title="CFB Power - College Football Ratings & Forecasting",
        page_icon="🏈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🏈 College Football Power Rankings & Matchup Forecaster")
    st.caption(
        "Opponent-adjusted Ridge ratings · situational efficiency analytics · weekly game forecasts"
    )

    # Load power rankings and team names
    rankings_obj, rankings_df, team_names = get_live_power_rankings(data_dir)
    if not team_names:
        team_names = resolve_team_names(data_dir)

    rankings_tab, matchup_tab, forecasts_tab, results_tab, update_tab, settings_tab = st.tabs(
        [
            "🏆 Power Rankings",
            "⚔️ Matchup Analyzer",
            "📈 Weekly Forecasts",
            "📊 Results Report",
            "🔄 Ingestion & Revisions",
            "⚙️ Settings",
        ]
    )

    busy = any(
        name in st.session_state and st.session_state[name][0].is_alive()
        for name in ("job", "report_job")
    )

    with rankings_tab:
        render_power_rankings_tab(rankings_obj, rankings_df)

    with matchup_tab:
        render_matchup_analyzer_tab(rankings_obj, rankings_df)

    with forecasts_tab:
        st.subheader("📈 Weekly Matchup Forecasts & Historical Runs")
        runs = []
        for path in sorted((data_dir / "runs").glob("*/predictions_*/manifest.json"), reverse=True):
            try:
                manifest = json.loads(path.read_text())
                if manifest.get("status") == "ready":
                    runs.append((path.parent, manifest))
            except (OSError, ValueError):
                continue

        if not runs:
            st.info(
                "No published forecast runs yet. Run Update & forecast or predict-upcoming to generate forecasts."
            )
        else:
            selected = st.selectbox(
                "Select forecast run",
                range(len(runs)),
                format_func=lambda i: str(runs[i][0].relative_to(data_dir)),
            )
            run, manifest = runs[selected]

            for warning in manifest.get("warnings", []):
                st.warning(warning)

            from hashlib import sha256

            valid = all(
                (run / item["file"]).is_file()
                and sha256((run / item["file"]).read_bytes()).hexdigest() == item["sha256"]
                for item in manifest["outputs"]
            )
            if not valid:
                st.error("Run artifacts failed integrity verification; not displaying them.")
            else:
                raw_preds = pd.read_csv(run / "predictions.csv")
                preds = enrich_predictions_frame(raw_preds, team_names)

                # Slate Summary Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Games Forecasted", len(preds))
                avg_total = preds["projected_total"].mean() if "projected_total" in preds else 0.0
                m2.metric("Avg Projected Total", f"{avg_total:.1f} pts")
                avg_spread = (
                    preds["projected_spread_home"].abs().mean()
                    if "projected_spread_home" in preds
                    else 0.0
                )
                m3.metric("Avg Projected Margin", f"{avg_spread:.1f} pts")
                home_fav_pct = (
                    (preds["projected_spread_home"] > 0).mean()
                    if "projected_spread_home" in preds
                    else 0.5
                )
                m4.metric("Home Teams Favored", f"{home_fav_pct:.1%}")

                # Filtering
                f1, f2 = st.columns([3, 1])
                with f1:
                    filter_q = st.text_input("🔍 Filter games by team name", key="forecast_filter")
                with f2:
                    view_mode = st.radio("View", ["Table", "Cards"], horizontal=True)

                display_df = preds
                if filter_q.strip():
                    display_df = display_df[
                        display_df["home_team"].str.contains(filter_q, case=False, na=False)
                        | display_df["away_team"].str.contains(filter_q, case=False, na=False)
                    ]

                pred_config = {
                    "matchup": st.column_config.TextColumn("Matchup", width="medium"),
                    "kickoff_utc": st.column_config.TextColumn("Kickoff (UTC)"),
                    "home_team": st.column_config.TextColumn("Home Team"),
                    "away_team": st.column_config.TextColumn("Away Team"),
                    "home_expected_points": st.column_config.NumberColumn(
                        "Home Proj Pts", format="%.1f"
                    ),
                    "away_expected_points": st.column_config.NumberColumn(
                        "Away Proj Pts", format="%.1f"
                    ),
                    "projected_spread_home": st.column_config.NumberColumn(
                        "Spread (Home)", format="%+.1f", help="Positive favors home team"
                    ),
                    "projected_total": st.column_config.NumberColumn("Total", format="%.1f"),
                    "home_win_probability": st.column_config.ProgressColumn(
                        "Home Win %", format="%.1f%%", min_value=0.0, max_value=1.0
                    ),
                    "low_history": st.column_config.CheckboxColumn("Low History"),
                }

                if view_mode == "Table":
                    cols_to_show = [c for c in pred_config.keys() if c in display_df.columns]
                    st.dataframe(
                        display_df[cols_to_show],
                        column_config=pred_config,
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    # Visual Card View
                    st.markdown("#### Game Forecast Cards")
                    for idx in range(0, len(display_df), 2):
                        row_cols = st.columns(2)
                        for col_idx, (_, game) in enumerate(
                            display_df.iloc[idx : idx + 2].iterrows()
                        ):
                            with row_cols[col_idx]:
                                home_name = game.get("home_team", "Home")
                                away_name = game.get("away_team", "Away")
                                spread_val = game.get("projected_spread_home", 0.0)
                                total_val = game.get("projected_total", 0.0)
                                prob_val = game.get("home_win_probability", 0.5)
                                home_pts = game.get("home_expected_points", 0.0)
                                away_pts = game.get("away_expected_points", 0.0)
                                kickoff = game.get("kickoff_utc", "")

                                favored_str = (
                                    f"{home_name} -{spread_val:.1f}"
                                    if spread_val > 0
                                    else f"{away_name} -{-spread_val:.1f}"
                                )

                                st.markdown(
                                    f"""
                                    <div style="border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px; margin-bottom: 12px; background: #FAFAFA;">
                                        <div style="font-size: 11px; color: #64748B; margin-bottom: 4px;">📅 {kickoff}</div>
                                        <div style="font-size: 16px; font-weight: bold; margin-bottom: 6px;">
                                            {away_name} <span style="color: #64748B; font-weight: normal;">@</span> {home_name}
                                        </div>
                                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                            <div><strong>Proj Score:</strong> {away_pts:.0f} - {home_pts:.0f}</div>
                                            <div><strong>Line:</strong> {favored_str}</div>
                                            <div><strong>O/U:</strong> {total_val:.1f}</div>
                                        </div>
                                        <div style="font-size: 12px; color: #475569;">
                                            Win Probability: {home_name} <strong>{prob_val:.1%}</strong> · {away_name} <strong>{1.0 - prob_val:.1%}</strong>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                with st.expander("Skipped games & exclusions"):
                    st.dataframe(pd.read_csv(run / "skipped.csv"), use_container_width=True)

                st.markdown("---")
                dl1, dl2, dl3 = st.columns(3)
                with dl1:
                    st.download_button(
                        "📥 Download predictions.csv",
                        preds.to_csv(index=False),
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
                with dl2:
                    st.download_button(
                        "📥 Download skipped.csv",
                        (run / "skipped.csv").read_bytes(),
                        file_name="skipped.csv",
                    )
                with dl3:
                    if (run / "upcoming_predictions.xlsx").is_file():
                        st.download_button(
                            "📥 Download upcoming_predictions.xlsx",
                            (run / "upcoming_predictions.xlsx").read_bytes(),
                            file_name="upcoming_predictions.xlsx",
                        )

    with results_tab:
        render_results(root, data_dir, busy)

    with update_tab:
        st.subheader("🔄 Ingestion & Revisions")
        st.write(
            "Choose every week to refresh, including weeks with missing results. All captures must succeed before assembly starts."
        )
        season = int(
            st.number_input("Season", min_value=2000, max_value=2100, value=2026, disabled=busy)
        )
        weeks = st.multiselect(
            "Weeks to capture", list(range(0, 21)), default=[6, 7], disabled=busy
        )
        st.info(
            "Each week uses one schedule request plus one box-score request per completed game. Repeated runs repeat requests. Partial successful snapshots remain available if a later step fails."
        )
        if st.button("Update & forecast", disabled=busy or not weeks):
            events = queue.Queue()
            secret = read_key(root)

            def worker():
                try:
                    run = refresh(
                        data_dir=data_dir,
                        season=season,
                        weeks=weeks,
                        key=secret,
                        emit=lambda message: events.put(("log", message)),
                        capture=ingest_cfbd_batch,
                        discover=discover_snapshots,
                        assemble=assemble_or_review,
                        forecast=predict_upcoming,
                    )
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
                        st.session_state.messages.append(
                            ("log", "Capture complete. Review revisions below before assembly.")
                        )
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
                st.caption(
                    "Working. Keep this server running; closing the app process interrupts the job."
                )
                time.sleep(1)
                st.rerun()
        render_review(data_dir, busy)

    with settings_tab:
        st.subheader("⚙️ Settings & API Configuration")
        st.warning(
            "The key is stored as plaintext in your local .env, owner-readable only. Never share that file."
        )
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
                st.error(
                    "Key test failed. Check the saved key, connection, and CFBD account quota."
                )


if __name__ == "__main__":
    main()
