"""Revision review widgets, rendered on the Streamlit main thread."""

import queue
import threading


def render_review(data_dir, busy):
    import pandas as pd
    import streamlit as st

    from .dataset_workflow import discover_snapshots
    from .revisions import inspect_snapshots, resume_review

    if st.button("Review existing snapshots (no API requests)", disabled=busy):
        try:
            st.session_state.review_plan = inspect_snapshots(discover_snapshots(data_dir))
        except Exception as exc:
            st.error(str(exc))
    plan = st.session_state.get("review_plan")
    if not plan:
        return
    st.subheader("Review data revisions")
    if not plan["conflicts"]:
        st.info("No conflicting versions found. Use Update & forecast to refresh data.")
        return
    st.warning(
        "Choose one complete game version for EVERY conflict. No version is accepted automatically. Approval applies only to this build; original snapshots are retained."
    )
    decisions = {}
    for game_id, versions in plan["conflicts"].items():
        with st.expander(f"Game {game_id}: {len(versions)} candidate snapshots", expanded=True):
            flattened = []
            for version in versions:
                flat = {f"game.{k}": value for k, value in version["bundle"]["game"].items()}
                for team in version["bundle"]["teams"]:
                    flat.update({f"team.{team['team_id']}.{k}": value for k, value in team.items()})
                flattened.append(flat)
            rows = []
            for field in sorted({key for flat in flattened for key in flat}):
                values = [flat.get(field) for flat in flattened]
                if any(value != values[0] for value in values[1:]):
                    rows.append(
                        {
                            "field": field,
                            **{v["run_id"]: value for v, value in zip(versions, values, strict=True)},
                        }
                    )
            st.dataframe(pd.DataFrame(rows), width="stretch")
            if st.checkbox(
                "Show complete versions and source hashes", key=f"details-{plan['token']}-{game_id}"
            ):
                st.json(versions)
            choices = [v["run_id"] for v in versions]
            picked = st.selectbox(
                "Version to use",
                [None, *choices],
                format_func=lambda value: "Select a version" if value is None else value,
                key=f"revision-{plan['token']}-{game_id}",
                disabled=busy,
            )
            if picked:
                decisions[game_id] = picked
    approved = st.checkbox(
        "I reviewed the differences and approve these complete game versions.",
        key=f"approve-{plan['token']}",
        disabled=busy,
    )
    if st.button(
        "Approve selected revisions & build forecast",
        disabled=busy or not approved or len(decisions) != len(plan["conflicts"]),
    ):
        events = queue.Queue()
        token, selected = plan["token"], dict(decisions)

        def worker():
            try:
                events.put(("log", "Re-verifying approved inputs; no API requests"))
                run = resume_review(data_dir, selected, token)
                events.put(("done", str(run)))
            except Exception as exc:
                events.put(("error", str(exc)))

        thread = threading.Thread(target=worker, daemon=True)
        st.session_state.job = (thread, events)
        st.session_state.messages = []
        del st.session_state.review_plan
        thread.start()
        st.rerun()
