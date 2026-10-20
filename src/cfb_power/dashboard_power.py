"""UI components and data loaders for Power Rankings and Matchup Analyzer in Streamlit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .power_ranking import MatchupComparison, PowerRankings, fit_power_rankings
from .team_names import resolve_team_names


@st.cache_data(show_spinner=False)
def load_cached_power_rankings_data(
    data_dir_str: str,
) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    """Load active games and stats, fit power rankings, and cache the serialized table."""
    data_dir = Path(data_dir_str)
    processed_dir = data_dir / "processed"
    games_csv = processed_dir / "games.csv"
    stats_csv = processed_dir / "team_game_stats.csv"

    if not games_csv.is_file() or not stats_csv.is_file():
        # Fallback to latest dataset
        datasets_dir = data_dir / "datasets"
        if datasets_dir.is_dir():
            candidates = sorted(
                [d for d in datasets_dir.iterdir() if d.is_dir() and (d / "games.csv").is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                games_csv = candidates[0] / "games.csv"
                stats_csv = candidates[0] / "team_game_stats.csv"

    if not games_csv.is_file() or not stats_csv.is_file():
        return None, None

    try:
        games = pd.read_csv(games_csv)
        stats = pd.read_csv(stats_csv)
        names = resolve_team_names(data_dir)
        rankings = fit_power_rankings(games, stats, team_names=names)
        df = rankings.to_dataframe()
        return df, {"rankings_object": rankings, "team_names": names}
    except Exception:
        return None, None


def get_live_power_rankings(
    data_dir: Path,
) -> tuple[PowerRankings | None, pd.DataFrame | None, dict[str | int, str]]:
    """Retrieve PowerRankings object and DataFrame."""
    df, meta = load_cached_power_rankings_data(str(data_dir))
    if meta is None or "rankings_object" not in meta:
        # Re-compute if needed
        processed_dir = data_dir / "processed"
        games_csv = processed_dir / "games.csv"
        stats_csv = processed_dir / "team_game_stats.csv"
        if not games_csv.is_file() or not stats_csv.is_file():
            return None, None, {}
        games = pd.read_csv(games_csv)
        stats = pd.read_csv(stats_csv)
        names = resolve_team_names(data_dir)
        rankings = fit_power_rankings(games, stats, team_names=names)
        return rankings, rankings.to_dataframe(), names
    return meta["rankings_object"], df, meta.get("team_names", {})


def render_power_rankings_tab(
    rankings: PowerRankings | None, rankings_df: pd.DataFrame | None
) -> None:
    """Render the polished Power Rankings tab."""
    st.subheader("🏆 Opponent-Adjusted Power Rankings")
    st.caption(
        "Composite ratings synthesize Ridge regression scoring margins, game-control efficiency, "
        "SOS schedule adjustments, and exponential recency weighting."
    )

    if rankings_df is None or rankings_df.empty:
        st.info(
            "No active games dataset found. Ingest games or activate a dataset to view power rankings."
        )
        return

    # Top KPI Metrics Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    best_overall = rankings_df.iloc[0]
    best_offense = rankings_df.sort_values("offensive_rating", ascending=False).iloc[0]
    best_defense = rankings_df.sort_values("defensive_rating", ascending=False).iloc[0]
    toughest_sos = rankings_df.sort_values("strength_of_schedule", ascending=False).iloc[0]

    kpi1.metric(
        "#1 Overall Team",
        f"{best_overall['team_name']}",
        f"{best_overall['power_rating']:+.1f} rating",
    )
    kpi2.metric(
        "Top Offense",
        f"{best_offense['team_name']}",
        f"{best_offense['offensive_rating']:+.1f} pts/G",
    )
    kpi3.metric(
        "Top Defense",
        f"{best_defense['team_name']}",
        f"{best_defense['defensive_rating']:+.1f} rating",
    )
    kpi4.metric(
        "Toughest Schedule",
        f"{toughest_sos['team_name']}",
        f"{toughest_sos['strength_of_schedule']:+.1f} SOS",
    )

    # Filter Controls
    fc1, fc2 = st.columns([2, 1])
    with fc1:
        search_query = st.text_input(
            "🔍 Search team name", placeholder="e.g. Ohio State, Georgia, Alabama..."
        )
    with fc2:
        available_tiers = list(rankings_df["tier"].unique())
        selected_tiers = st.multiselect("Filter by Tier", available_tiers, default=available_tiers)

    filtered_df = rankings_df[rankings_df["tier"].isin(selected_tiers)]
    if search_query.strip():
        filtered_df = filtered_df[
            filtered_df["team_name"].str.contains(search_query.strip(), case=False, na=False)
        ]

    # Column formatting config
    column_config = {
        "rank": st.column_config.NumberColumn("Rank", format="#%d", width="small"),
        "team_name": st.column_config.TextColumn("Team", width="medium"),
        "tier": st.column_config.TextColumn("Tier", width="small"),
        "record": st.column_config.TextColumn("Record", width="small"),
        "power_rating": st.column_config.NumberColumn(
            "Overall Rating",
            format="%+.2f",
            help="Composite power rating (+/- points vs average FBS team)",
        ),
        "offensive_rating": st.column_config.NumberColumn(
            "Offense", format="%+.2f", help="Points scored relative to average opponent"
        ),
        "defensive_rating": st.column_config.NumberColumn(
            "Defense", format="%+.2f", help="Defensive point suppression (+ is better)"
        ),
        "strength_of_schedule": st.column_config.NumberColumn(
            "SOS", format="%+.2f", help="Strength of Schedule rating (+ is harder)"
        ),
        "recent_form_rating": st.column_config.NumberColumn(
            "Recent Form", format="%+.2f", help="Recency-weighted 3-game trajectory"
        ),
        "yards_per_play": st.column_config.NumberColumn("Yds/Play Off", format="%.2f"),
        "yards_per_play_allowed": st.column_config.NumberColumn("Yds/Play Def", format="%.2f"),
        "third_down_rate": st.column_config.NumberColumn("3rd Down %", format="%.1f%%"),
        "third_down_rate_allowed": st.column_config.NumberColumn("3rd Down Def %", format="%.1f%%"),
        "red_zone_td_rate": st.column_config.NumberColumn("RZ TD %", format="%.1f%%"),
        "red_zone_td_rate_allowed": st.column_config.NumberColumn("RZ TD Def %", format="%.1f%%"),
        "turnover_margin_pg": st.column_config.NumberColumn("TO Margin/G", format="%+.2f"),
        "time_of_possession_min": st.column_config.NumberColumn("TOP (Min)", format="%.1f"),
    }

    display_cols = [c for c in column_config.keys() if c in filtered_df.columns]
    st.dataframe(
        filtered_df[display_cols],
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "📥 Download Power Rankings (CSV)",
        filtered_df.to_csv(index=False),
        file_name="cfb_power_rankings.csv",
        mime="text/csv",
    )


def render_matchup_analyzer_tab(
    rankings: PowerRankings | None, rankings_df: pd.DataFrame | None
) -> None:
    """Render the interactive head-to-head matchup analyzer."""
    st.subheader("⚔️ Head-to-Head Matchup Analyzer")
    st.caption(
        "Compare any two FBS teams using opponent-adjusted ratings, situational metrics, and efficiency splits."
    )

    if rankings is None or rankings_df is None or rankings_df.empty:
        st.info(
            "Power rankings data is required for matchup analysis. Please assemble games first."
        )
        return

    # Sort teams alphabetically by name
    sorted_teams = rankings_df.sort_values("team_name")
    team_options = {row["team_name"]: row["team_id"] for _, row in sorted_teams.iterrows()}
    team_names_list = list(team_options.keys())

    default_team_a_idx = 0
    default_team_b_idx = min(1, len(team_names_list) - 1)
    if "Notre Dame" in team_names_list:
        default_team_a_idx = team_names_list.index("Notre Dame")
    if "Ohio State" in team_names_list:
        default_team_b_idx = team_names_list.index("Ohio State")

    col_a, col_vs, col_b = st.columns([5, 1, 5])
    with col_a:
        team_a_name = st.selectbox(
            "Team A (Home / Favorite)", team_names_list, index=default_team_a_idx
        )
    with col_vs:
        st.markdown(
            "<h3 style='text-align: center; margin-top: 25px;'>VS</h3>", unsafe_allow_html=True
        )
    with col_b:
        team_b_name = st.selectbox(
            "Team B (Away / Underdog)", team_names_list, index=default_team_b_idx
        )

    team_a_id = team_options[team_a_name]
    team_b_id = team_options[team_b_name]

    # Settings and Adjustments
    with st.expander("⚙️ Matchup Settings & Key Player / QB Adjustments", expanded=False):
        set_col1, set_col2, set_col3 = st.columns(3)
        with set_col1:
            neutral_site = st.checkbox("Neutral Site Game (No HFA)", value=False)
        with set_col2:
            adj_a = st.slider(f"{team_a_name} QB/Roster Adj (pts)", -7.0, 7.0, 0.0, 0.5)
        with set_col3:
            adj_b = st.slider(f"{team_b_name} QB/Roster Adj (pts)", -7.0, 7.0, 0.0, 0.5)

    if team_a_id == team_b_id:
        st.warning("Select two different teams to generate a matchup comparison.")
        return

    try:
        comparison: MatchupComparison = rankings.compare_teams(
            team_a_id, team_b_id, neutral_site=neutral_site, team_a_adj=adj_a, team_b_adj=adj_b
        )
    except Exception as exc:
        st.error(f"Error computing matchup: {exc}")
        return

    favored = comparison.favored_team_name
    spread = comparison.projected_spread
    win_prob = (
        comparison.team_a_win_probability
        if comparison.favored_team == team_a_id
        else comparison.team_b_win_probability
    )

    # Matchup Outcome Banner
    st.markdown("---")
    banner1, banner2, banner3, banner4 = st.columns(4)
    banner1.metric(
        "Projected Winner",
        f"🏆 {favored}",
        f"-{spread:.1f} pts spread",
    )
    banner2.metric(
        "Projected Score",
        f"{team_a_name} {comparison.team_a_expected_points:.0f} - {comparison.team_b_expected_points:.0f} {team_b_name}",
        f"{team_a_name} ({comparison.team_a_expected_points:.1f}) · {team_b_name} ({comparison.team_b_expected_points:.1f})",
    )
    banner3.metric(
        "Projected Total",
        f"{comparison.projected_total:.1f} pts",
        "Over / Under line",
    )
    banner4.metric(
        "Win Probability",
        f"{favored} {win_prob:.1%}",
        f"Underdog: {1.0 - win_prob:.1%}",
    )

    # Win Probability Progress Bar
    prob_a = comparison.team_a_win_probability
    st.caption(
        f"Win Probability Split: **{team_a_name} {prob_a:.1%}** vs **{team_b_name} {1.0 - prob_a:.1%}**"
    )
    st.progress(prob_a)

    # Tale of the Tape Side-by-Side Comparison
    st.subheader("📋 Tale of the Tape (Situational & Efficiency Comparison)")

    prof_a = comparison.team_a_profile
    prof_b = comparison.team_b_profile

    tape_data = [
        {
            "Metric": "Power Rating",
            team_a_name: f"{prof_a.power_rating:+.2f}",
            "Statistical Edge": f"👑 {team_a_name} (+{prof_a.power_rating - prof_b.power_rating:.2f})"
            if prof_a.power_rating > prof_b.power_rating
            else f"👑 {team_b_name} (+{prof_b.power_rating - prof_a.power_rating:.2f})",
            team_b_name: f"{prof_b.power_rating:+.2f}",
        },
        {
            "Metric": "Offensive Rating",
            team_a_name: f"{prof_a.offensive_rating:+.2f}",
            "Statistical Edge": f"👑 {team_a_name}"
            if prof_a.offensive_rating > prof_b.offensive_rating
            else f"👑 {team_b_name}",
            team_b_name: f"{prof_b.offensive_rating:+.2f}",
        },
        {
            "Metric": "Defensive Rating",
            team_a_name: f"{prof_a.defensive_rating:+.2f}",
            "Statistical Edge": f"👑 {team_a_name}"
            if prof_a.defensive_rating > prof_b.defensive_rating
            else f"👑 {team_b_name}",
            team_b_name: f"{prof_b.defensive_rating:+.2f}",
        },
        {
            "Metric": "Strength of Schedule",
            team_a_name: f"{prof_a.strength_of_schedule:+.2f}",
            "Statistical Edge": f"Harder: {team_a_name}"
            if prof_a.strength_of_schedule > prof_b.strength_of_schedule
            else f"Harder: {team_b_name}",
            team_b_name: f"{prof_b.strength_of_schedule:+.2f}",
        },
        {
            "Metric": "Recent Form (L3)",
            team_a_name: f"{prof_a.recent_form_rating:+.2f}",
            "Statistical Edge": f"👑 {team_a_name}"
            if prof_a.recent_form_rating > prof_b.recent_form_rating
            else f"👑 {team_b_name}",
            team_b_name: f"{prof_b.recent_form_rating:+.2f}",
        },
        {
            "Metric": "Yards / Play (Offense)",
            team_a_name: f"{prof_a.yards_per_play:.2f}",
            "Statistical Edge": f"👑 {team_a_name}"
            if prof_a.yards_per_play > prof_b.yards_per_play
            else f"👑 {team_b_name}",
            team_b_name: f"{prof_b.yards_per_play:.2f}",
        },
        {
            "Metric": "Yards / Play Allowed (Defense)",
            team_a_name: f"{prof_a.yards_per_play_allowed:.2f}",
            "Statistical Edge": f"👑 {team_a_name}"
            if prof_a.yards_per_play_allowed < prof_b.yards_per_play_allowed
            else f"👑 {team_b_name}",
            team_b_name: f"{prof_b.yards_per_play_allowed:.2f}",
        },
        {
            "Metric": "3rd Down Conversion %",
            team_a_name: f"{prof_a.third_down_rate:.1%}",
            "Statistical Edge": f"👑 {team_a_name}"
            if prof_a.third_down_rate > prof_b.third_down_rate
            else f"👑 {team_b_name}",
            team_b_name: f"{prof_b.third_down_rate:.1%}",
        },
        {
            "Metric": "Red Zone TD %",
            team_a_name: f"{prof_a.red_zone_td_rate:.1%}",
            "Statistical Edge": f"👑 {team_a_name}"
            if prof_a.red_zone_td_rate > prof_b.red_zone_td_rate
            else f"👑 {team_b_name}",
            team_b_name: f"{prof_b.red_zone_td_rate:.1%}",
        },
        {
            "Metric": "Turnover Margin / Game",
            team_a_name: f"{prof_a.turnover_margin_pg:+.2f}",
            "Statistical Edge": f"👑 {team_a_name}"
            if prof_a.turnover_margin_pg > prof_b.turnover_margin_pg
            else f"👑 {team_b_name}",
            team_b_name: f"{prof_b.turnover_margin_pg:+.2f}",
        },
        {
            "Metric": "Penalty Yards / Game",
            team_a_name: f"{prof_a.penalty_yards_pg:.1f}",
            "Statistical Edge": f"👑 {team_a_name}"
            if prof_a.penalty_yards_pg < prof_b.penalty_yards_pg
            else f"👑 {team_b_name}",
            team_b_name: f"{prof_b.penalty_yards_pg:.1f}",
        },
        {
            "Metric": "Time of Possession",
            team_a_name: f"{prof_a.time_of_possession_min:.1f} min",
            "Statistical Edge": f"👑 {team_a_name}"
            if prof_a.time_of_possession_min > prof_b.time_of_possession_min
            else f"👑 {team_b_name}",
            team_b_name: f"{prof_b.time_of_possession_min:.1f} min",
        },
    ]

    tape_df = pd.DataFrame(tape_data)
    st.table(tape_df)
