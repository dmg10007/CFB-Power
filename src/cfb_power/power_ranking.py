"""Opponent-adjusted Power Rankings, statistical profiling, and head-to-head comparison engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from .model import _normal_cdf
from .schemas import PREDICTION_COLUMNS
from .validation import validate_games, validate_team_game_stats

DEFAULT_HOME_ADVANTAGE = 2.5
DEFAULT_LEAGUE_POINTS = 27.0
DEFAULT_MARGIN_STD = 14.0
DEFAULT_RECENCY_DECAY = 0.94
DEFAULT_PRIOR_WEIGHT = 3.0


@dataclass
class TeamPowerProfile:
    """Complete power rating and statistical efficiency profile for a team."""

    team_id: str | int
    team_name: str | None = None
    rank: int = 0
    tier: str = "Average"
    power_rating: float = 0.0
    offensive_rating: float = 0.0
    defensive_rating: float = 0.0
    strength_of_schedule: float = 0.0
    recent_form_rating: float = 0.0
    prior_rating: float = 0.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0

    # Efficiency statistics
    yards_per_play: float = 0.0
    yards_per_play_allowed: float = 0.0
    passing_yards_per_play: float = 0.0
    rushing_yards_per_play: float = 0.0
    third_down_rate: float = 0.0
    third_down_rate_allowed: float = 0.0
    red_zone_td_rate: float = 0.0
    red_zone_td_rate_allowed: float = 0.0
    turnovers_pg: float = 0.0
    takeaways_pg: float = 0.0
    turnover_margin_pg: float = 0.0
    penalties_pg: float = 0.0
    penalty_yards_pg: float = 0.0
    time_of_possession_sec: float = 0.0

    @property
    def record(self) -> str:
        if self.ties > 0:
            return f"{self.wins}-{self.losses}-{self.ties}"
        return f"{self.wins}-{self.losses}"

    @property
    def display_name(self) -> str:
        return self.team_name if self.team_name else str(self.team_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "team_id": self.team_id,
            "team_name": self.display_name,
            "tier": self.tier,
            "record": self.record,
            "power_rating": round(self.power_rating, 2),
            "offensive_rating": round(self.offensive_rating, 2),
            "defensive_rating": round(self.defensive_rating, 2),
            "strength_of_schedule": round(self.strength_of_schedule, 2),
            "recent_form_rating": round(self.recent_form_rating, 2),
            "games_played": self.games_played,
            "yards_per_play": round(self.yards_per_play, 2),
            "yards_per_play_allowed": round(self.yards_per_play_allowed, 2),
            "passing_yards_per_play": round(self.passing_yards_per_play, 2),
            "rushing_yards_per_play": round(self.rushing_yards_per_play, 2),
            "third_down_rate": round(self.third_down_rate, 3),
            "third_down_rate_allowed": round(self.third_down_rate_allowed, 3),
            "red_zone_td_rate": round(self.red_zone_td_rate, 3),
            "red_zone_td_rate_allowed": round(self.red_zone_td_rate_allowed, 3),
            "turnover_margin_pg": round(self.turnover_margin_pg, 2),
            "penalties_pg": round(self.penalties_pg, 1),
            "penalty_yards_pg": round(self.penalty_yards_pg, 1),
            "time_of_possession_min": round(self.time_of_possession_sec / 60.0, 1),
        }


@dataclass
class MatchupComparison:
    """Head-to-head comparison and projection between two teams."""

    team_a: TeamPowerProfile
    team_b: TeamPowerProfile
    neutral_site: bool
    home_team: str | int | None
    team_a_expected_points: float
    team_b_expected_points: float
    projected_spread: float  # Positive favors Team A, negative favors Team B
    projected_total: float
    team_a_win_probability: float
    team_b_win_probability: float
    team_a_score_low: float
    team_a_score_high: float
    team_b_score_low: float
    team_b_score_high: float
    confidence_tier: str
    edge_summary: list[str] = field(default_factory=list)
    stat_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def favored_team(self) -> str:
        if abs(self.projected_spread) < 0.2:
            return "Even / Pick'em"
        if self.projected_spread > 0:
            return f"{self.team_a.display_name} -{abs(self.projected_spread):.1f}"
        return f"{self.team_b.display_name} -{abs(self.projected_spread):.1f}"


@dataclass
class PowerRankings:
    """Collection of power-ranked teams with comparison and forecast tools."""

    season: int
    teams: dict[str | int, TeamPowerProfile]
    home_advantage: float = DEFAULT_HOME_ADVANTAGE
    league_points: float = DEFAULT_LEAGUE_POINTS
    margin_std: float = DEFAULT_MARGIN_STD

    def get_team(self, team_id: str | int) -> TeamPowerProfile:
        if team_id in self.teams:
            return self.teams[team_id]
        # Match by string representation or display name
        for team in self.teams.values():
            if str(team.team_id) == str(team_id) or (
                team.team_name and team.team_name.lower() == str(team_id).lower()
            ):
                return team
        raise KeyError(f"Team '{team_id}' not found in power rankings.")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all team profiles to a sorted pandas DataFrame."""
        rows = [profile.to_dict() for profile in self.teams.values()]
        return pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)

    def compare_teams(
        self,
        team_a_id: str | int,
        team_b_id: str | int,
        neutral_site: bool = False,
        home_team: str | int | None = None,
        roster_adjustments: dict[str | int, float] | None = None,
    ) -> MatchupComparison:
        """Compare any two teams and forecast score, spread, and statistical edges."""
        team_a = self.get_team(team_a_id)
        team_b = self.get_team(team_b_id)

        # Determine home-field bonus
        hfa_a = 0.0
        hfa_b = 0.0
        if not neutral_site:
            if home_team is not None:
                if str(home_team) == str(team_a.team_id) or home_team == team_a.display_name:
                    hfa_a = self.home_advantage
                elif str(home_team) == str(team_b.team_id) or home_team == team_b.display_name:
                    hfa_b = self.home_advantage
            else:
                # Default: Team A is considered home if home_team not specified and not neutral
                hfa_a = self.home_advantage

        # Apply optional player / QB / roster availability adjustments
        adj_a = (roster_adjustments or {}).get(team_a.team_id, 0.0)
        adj_b = (roster_adjustments or {}).get(team_b.team_id, 0.0)

        # Expected scores: League baseline + team offense - opponent defense + HFA/adjustments
        exp_a = max(
            3.0,
            round(
                self.league_points
                + team_a.offensive_rating
                - team_b.defensive_rating
                + (hfa_a - hfa_b) / 2.0
                + adj_a,
                1,
            ),
        )
        exp_b = max(
            3.0,
            round(
                self.league_points
                + team_b.offensive_rating
                - team_a.defensive_rating
                + (hfa_b - hfa_a) / 2.0
                + adj_b,
                1,
            ),
        )

        projected_spread = round((exp_a - exp_b), 1)
        projected_total = round((exp_a + exp_b), 1)

        z = projected_spread / self.margin_std
        win_prob_a = float(np.clip(_normal_cdf(np.array([z]))[0], 0.01, 0.99))
        win_prob_b = float(np.clip(1.0 - win_prob_a, 0.01, 0.99))

        band = self.margin_std / sqrt(2.0)
        score_low_a = max(0.0, round(exp_a - band, 1))
        score_high_a = round(exp_a + band, 1)
        score_low_b = max(0.0, round(exp_b - band, 1))
        score_high_b = round(exp_b + band, 1)

        prob_diff = abs(win_prob_a - 0.5) * 2
        confidence_tier = "High" if prob_diff >= 0.35 else "Medium" if prob_diff >= 0.15 else "Low"

        # Edge analysis
        edges: list[str] = []
        if team_a.power_rating > team_b.power_rating + 5.0:
            edges.append(
                f"Overall Rating: {team_a.display_name} (+{team_a.power_rating - team_b.power_rating:.1f} pts)"
            )
        elif team_b.power_rating > team_a.power_rating + 5.0:
            edges.append(
                f"Overall Rating: {team_b.display_name} (+{team_b.power_rating - team_a.power_rating:.1f} pts)"
            )

        # Yards per play differential
        ypp_diff_a = team_a.yards_per_play - team_b.yards_per_play_allowed
        ypp_diff_b = team_b.yards_per_play - team_a.yards_per_play_allowed
        if ypp_diff_a > 0.6:
            edges.append(
                f"Offensive Efficiency: {team_a.display_name} gains {team_a.yards_per_play:.2f} YPP vs {team_b.display_name} defense allowing {team_b.yards_per_play_allowed:.2f}"
            )
        if ypp_diff_b > 0.6:
            edges.append(
                f"Offensive Efficiency: {team_b.display_name} gains {team_b.yards_per_play:.2f} YPP vs {team_a.display_name} defense allowing {team_a.yards_per_play_allowed:.2f}"
            )

        # Third down conversion edge
        if team_a.third_down_rate > team_b.third_down_rate_allowed + 0.08:
            edges.append(
                f"3rd Down: {team_a.display_name} converts {team_a.third_down_rate:.1%} vs {team_b.display_name} allowing {team_b.third_down_rate_allowed:.1%}"
            )
        if team_b.third_down_rate > team_a.third_down_rate_allowed + 0.08:
            edges.append(
                f"3rd Down: {team_b.display_name} converts {team_b.third_down_rate:.1%} vs {team_a.display_name} allowing {team_a.third_down_rate_allowed:.1%}"
            )

        # Turnover edge
        to_diff = team_a.turnover_margin_pg - team_b.turnover_margin_pg
        if to_diff >= 0.7:
            edges.append(
                f"Turnover Margin: {team_a.display_name} ({team_a.turnover_margin_pg:+.1f}/gm) vs {team_b.display_name} ({team_b.turnover_margin_pg:+.1f}/gm)"
            )
        elif to_diff <= -0.7:
            edges.append(
                f"Turnover Margin: {team_b.display_name} ({team_b.turnover_margin_pg:+.1f}/gm) vs {team_a.display_name} ({team_a.turnover_margin_pg:+.1f}/gm)"
            )

        # Build side-by-side comparison table
        stat_table = pd.DataFrame(
            [
                {
                    "Metric": "Power Rank",
                    team_a.display_name: f"#{team_a.rank}",
                    team_b.display_name: f"#{team_b.rank}",
                },
                {
                    "Metric": "Power Rating (Net pts)",
                    team_a.display_name: f"{team_a.power_rating:+.1f}",
                    team_b.display_name: f"{team_b.power_rating:+.1f}",
                },
                {
                    "Metric": "Tier",
                    team_a.display_name: team_a.tier,
                    team_b.display_name: team_b.tier,
                },
                {
                    "Metric": "Record",
                    team_a.display_name: team_a.record,
                    team_b.display_name: team_b.record,
                },
                {
                    "Metric": "Offensive Rating",
                    team_a.display_name: f"{team_a.offensive_rating:+.1f}",
                    team_b.display_name: f"{team_b.offensive_rating:+.1f}",
                },
                {
                    "Metric": "Defensive Rating",
                    team_a.display_name: f"{team_a.defensive_rating:+.1f}",
                    team_b.display_name: f"{team_b.defensive_rating:+.1f}",
                },
                {
                    "Metric": "Strength of Schedule",
                    team_a.display_name: f"{team_a.strength_of_schedule:+.1f}",
                    team_b.display_name: f"{team_b.strength_of_schedule:+.1f}",
                },
                {
                    "Metric": "Recent Form Rating",
                    team_a.display_name: f"{team_a.recent_form_rating:+.1f}",
                    team_b.display_name: f"{team_b.recent_form_rating:+.1f}",
                },
                {
                    "Metric": "Yards / Play (Off / Def)",
                    team_a.display_name: f"{team_a.yards_per_play:.2f} / {team_a.yards_per_play_allowed:.2f}",
                    team_b.display_name: f"{team_b.yards_per_play:.2f} / {team_b.yards_per_play_allowed:.2f}",
                },
                {
                    "Metric": "Pass Yds / Play",
                    team_a.display_name: f"{team_a.passing_yards_per_play:.2f}",
                    team_b.display_name: f"{team_b.passing_yards_per_play:.2f}",
                },
                {
                    "Metric": "Rush Yds / Play",
                    team_a.display_name: f"{team_a.rushing_yards_per_play:.2f}",
                    team_b.display_name: f"{team_b.rushing_yards_per_play:.2f}",
                },
                {
                    "Metric": "3rd Down % (Off / Def)",
                    team_a.display_name: f"{team_a.third_down_rate:.1%} / {team_a.third_down_rate_allowed:.1%}",
                    team_b.display_name: f"{team_b.third_down_rate:.1%} / {team_b.third_down_rate_allowed:.1%}",
                },
                {
                    "Metric": "Red Zone TD % (Off / Def)",
                    team_a.display_name: f"{team_a.red_zone_td_rate:.1%} / {team_a.red_zone_td_rate_allowed:.1%}",
                    team_b.display_name: f"{team_b.red_zone_td_rate:.1%} / {team_b.red_zone_td_rate_allowed:.1%}",
                },
                {
                    "Metric": "Turnover Margin / Game",
                    team_a.display_name: f"{team_a.turnover_margin_pg:+.2f}",
                    team_b.display_name: f"{team_b.turnover_margin_pg:+.2f}",
                },
                {
                    "Metric": "Penalties / Penalty Yds",
                    team_a.display_name: f"{team_a.penalties_pg:.1f} / {team_a.penalty_yards_pg:.0f} yds",
                    team_b.display_name: f"{team_b.penalties_pg:.1f} / {team_b.penalty_yards_pg:.0f} yds",
                },
                {
                    "Metric": "Time of Possession",
                    team_a.display_name: f"{int(team_a.time_of_possession_sec // 60)}:{int(team_a.time_of_possession_sec % 60):02d}",
                    team_b.display_name: f"{int(team_b.time_of_possession_sec // 60)}:{int(team_b.time_of_possession_sec % 60):02d}",
                },
            ]
        )

        return MatchupComparison(
            team_a=team_a,
            team_b=team_b,
            neutral_site=neutral_site,
            home_team=home_team,
            team_a_expected_points=exp_a,
            team_b_expected_points=exp_b,
            projected_spread=projected_spread,
            projected_total=projected_total,
            team_a_win_probability=win_prob_a,
            team_b_win_probability=win_prob_b,
            team_a_score_low=score_low_a,
            team_a_score_high=score_high_a,
            team_b_score_low=score_low_b,
            team_b_score_high=score_high_b,
            confidence_tier=confidence_tier,
            edge_summary=edges,
            stat_comparison=stat_table,
        )

    def predict_slate(
        self,
        fixtures: pd.DataFrame,
        roster_adjustments: dict[str | int, float] | None = None,
        model_version: str = "v1.2.0-power-rankings",
    ) -> pd.DataFrame:
        """Generate full predictions for a DataFrame of fixtures."""
        predictions: list[dict] = []
        for _, row in fixtures.iterrows():
            home_id = row["home_team_id"]
            away_id = row["away_team_id"]
            neutral = bool(row.get("neutral_site", False))

            try:
                comp = self.compare_teams(
                    team_a_id=home_id,
                    team_b_id=away_id,
                    neutral_site=neutral,
                    home_team=home_id,
                    roster_adjustments=roster_adjustments,
                )
                exp_home = comp.team_a_expected_points
                exp_away = comp.team_b_expected_points
                spread_home = comp.projected_spread
                total = comp.projected_total
                win_prob_home = comp.team_a_win_probability
                home_low = comp.team_a_score_low
                home_high = comp.team_a_score_high
                away_low = comp.team_b_score_low
                away_high = comp.team_b_score_high
                tier = comp.confidence_tier
            except KeyError:
                # If either team lacks history, fall back to league mean
                exp_home = self.league_points + (self.home_advantage / 2 if not neutral else 0.0)
                exp_away = self.league_points - (self.home_advantage / 2 if not neutral else 0.0)
                spread_home = exp_home - exp_away
                total = exp_home + exp_away
                win_prob_home = float(_normal_cdf(np.array([spread_home / self.margin_std]))[0])
                band = self.margin_std / sqrt(2.0)
                home_low, home_high = max(0.0, exp_home - band), exp_home + band
                away_low, away_high = max(0.0, exp_away - band), exp_away + band
                tier = "Low"

            record = {
                "game_id": row["game_id"],
                "season": row.get("season", self.season),
                "week": row.get("week", 1),
                "game_date": row.get("game_date", ""),
                "away_team_id": away_id,
                "home_team_id": home_id,
                "away_expected_points": round(exp_away, 1),
                "home_expected_points": round(exp_home, 1),
                "projected_spread_home": round(spread_home, 1),
                "projected_total": round(total, 1),
                "home_win_probability": round(win_prob_home, 4),
                "away_score_low": round(away_low, 1),
                "away_score_high": round(away_high, 1),
                "home_score_low": round(home_low, 1),
                "home_score_high": round(home_high, 1),
                "confidence_tier": tier,
                "model_version": model_version,
            }
            predictions.append(record)

        result_df = pd.DataFrame(predictions)
        # Ensure exact PREDICTION_COLUMNS ordering
        for col in PREDICTION_COLUMNS:
            if col not in result_df.columns:
                result_df[col] = np.nan
        return result_df[PREDICTION_COLUMNS]


def _assign_tier(rating: float) -> str:
    if rating >= 14.0:
        return "Tier 1: Elite"
    if rating >= 7.0:
        return "Tier 2: Contender"
    if rating >= 1.0:
        return "Tier 3: Above Average"
    if rating >= -5.0:
        return "Tier 4: Average"
    if rating >= -12.0:
        return "Tier 5: Below Average"
    return "Tier 6: Rebuilding"


def fit_power_rankings(
    games: pd.DataFrame,
    stats: pd.DataFrame,
    team_names: dict[str | int, str] | None = None,
    priors: dict[str | int, float] | None = None,
    talent_scores: dict[str | int, float] | None = None,
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
    recency_decay: float = DEFAULT_RECENCY_DECAY,
    prior_weight: float = DEFAULT_PRIOR_WEIGHT,
    iterations: int = 15,
) -> PowerRankings:
    """Compute opponent-adjusted offensive, defensive, and net power ratings."""
    games = validate_games(games)
    stats = validate_team_game_stats(stats, games)

    done = games["completed"].astype(bool)
    completed_games = games[done].copy()
    if completed_games.empty:
        raise ValueError("Cannot fit power rankings without completed games.")

    season = int(completed_games["season"].iloc[0])
    max_week = int(completed_games["week"].max())

    # Build per-team box game records
    merged = stats.merge(
        completed_games[["game_id", "season", "week", "neutral_site", "game_date"]],
        on="game_id",
        how="inner",
    )

    all_teams = sorted(set(merged["team_id"].unique()))

    # Calculate league points per game and margin standard deviation
    league_points = float(merged["points"].mean())
    margins = completed_games["home_score"] - completed_games["away_score"]
    margin_std = max(float(margins.std(ddof=1)), 8.0) if len(margins) > 1 else DEFAULT_MARGIN_STD

    # Pre-calculate game weights using recency decay
    # weight = decay ** (max_week - week)
    merged["recency_weight"] = recency_decay ** (max_week - merged["week"]).clip(lower=0)
    merged["recent_form_weight"] = 0.78 ** (max_week - merged["week"]).clip(lower=0)

    # Compile basic win/loss/tie and statistical efficiency metrics for each team
    stat_profiles: dict[str | int, dict[str, Any]] = {}
    for team_id in all_teams:
        team_rows = merged[merged["team_id"] == team_id]
        n_games = len(team_rows)
        if n_games == 0:
            continue

        wins = int((team_rows["points"] > team_rows["points_allowed"]).sum())
        losses = int((team_rows["points"] < team_rows["points_allowed"]).sum())
        ties = int((team_rows["points"] == team_rows["points_allowed"]).sum())

        tot_plays = float(team_rows["offensive_plays"].sum())
        tot_yds = float(team_rows["total_yards"].sum())
        tot_pass = float(team_rows["passing_yards"].sum())
        tot_rush = float(team_rows["rushing_yards"].sum())

        tot_3rd_att = float(team_rows["third_down_attempts"].sum())
        tot_3rd_conv = float(team_rows["third_down_conversions"].sum())

        tot_rz_att = float(team_rows["red_zone_attempts"].sum())
        tot_rz_td = float(team_rows["red_zone_touchdowns"].sum())

        tot_to = float(team_rows["turnovers"].sum())
        tot_tak = float(team_rows["takeaways"].sum())

        tot_pen = float(team_rows["penalties"].sum())
        tot_pen_yds = float(team_rows["penalty_yards"].sum())
        tot_top = float(team_rows["time_of_possession_seconds"].sum())

        # Opponent stats allowed
        opp_games = merged[
            (merged["opponent_id"] == team_id) & (merged["game_id"].isin(team_rows["game_id"]))
        ]
        opp_plays = float(opp_games["offensive_plays"].sum())
        opp_yds = float(opp_games["total_yards"].sum())
        opp_3rd_att = float(opp_games["third_down_attempts"].sum())
        opp_3rd_conv = float(opp_games["third_down_conversions"].sum())
        opp_rz_att = float(opp_games["red_zone_attempts"].sum())
        opp_rz_td = float(opp_games["red_zone_touchdowns"].sum())

        stat_profiles[team_id] = {
            "games_played": n_games,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "yards_per_play": tot_yds / tot_plays if tot_plays > 0 else 0.0,
            "yards_per_play_allowed": opp_yds / opp_plays if opp_plays > 0 else 0.0,
            "passing_yards_per_play": tot_pass / tot_plays if tot_plays > 0 else 0.0,
            "rushing_yards_per_play": tot_rush / tot_plays if tot_plays > 0 else 0.0,
            "third_down_rate": tot_3rd_conv / tot_3rd_att if tot_3rd_att > 0 else 0.0,
            "third_down_rate_allowed": opp_3rd_conv / opp_3rd_att if opp_3rd_att > 0 else 0.0,
            "red_zone_td_rate": tot_rz_td / tot_rz_att if tot_rz_att > 0 else 0.0,
            "red_zone_td_rate_allowed": opp_rz_td / opp_rz_att if opp_rz_att > 0 else 0.0,
            "turnovers_pg": tot_to / n_games,
            "takeaways_pg": tot_tak / n_games,
            "turnover_margin_pg": (tot_tak - tot_to) / n_games,
            "penalties_pg": tot_pen / n_games,
            "penalty_yards_pg": tot_pen_yds / n_games,
            "time_of_possession_sec": tot_top / n_games,
        }

    # Initialize ratings
    # Base prior from talent or prior ratings if provided
    team_priors: dict[str | int, float] = {}
    for team_id in all_teams:
        prior_val = 0.0
        if priors and team_id in priors:
            prior_val = priors[team_id]
        elif talent_scores and team_id in talent_scores:
            # Map talent score (e.g. 0-100 or 500-1000) to rating spread
            avg_talent = float(np.mean(list(talent_scores.values())))
            prior_val = (talent_scores[team_id] - avg_talent) * 0.25
        team_priors[team_id] = prior_val

    # Initialize offensive and defensive ratings
    off_ratings = {team: team_priors[team] / 2.0 for team in all_teams}
    def_ratings = {team: team_priors[team] / 2.0 for team in all_teams}

    # Iterative Opponent Adjustment Solver
    # Points = League_Mean + Offense_Team - Defense_Opponent + HFA
    ridge_lambda = 4.0
    for _ in range(iterations):
        new_off = {}
        new_def = {}
        for team in all_teams:
            rows = merged[merged["team_id"] == team]
            if len(rows) == 0:
                new_off[team] = off_ratings[team]
                new_def[team] = def_ratings[team]
                continue

            weights = rows["recency_weight"].to_numpy()
            pts = rows["points"].to_numpy()
            pts_allowed = rows["points_allowed"].to_numpy()
            is_neutral = rows["neutral_site"].astype(bool).to_numpy()
            is_home = rows["is_home"].astype(bool).to_numpy()

            hfa_term = np.where(
                is_neutral, 0.0, np.where(is_home, home_advantage / 2.0, -home_advantage / 2.0)
            )
            opps = rows["opponent_id"].tolist()

            opp_def = np.array([def_ratings.get(opp, 0.0) for opp in opps])
            opp_off = np.array([off_ratings.get(opp, 0.0) for opp in opps])

            # Offense: how much above (League + HFA - Opp_Def) did team score?
            off_numer = np.sum(weights * (pts - (league_points + hfa_term) + opp_def))
            # Defense: how much below (League - HFA + Opp_Off) did team allow?
            def_numer = np.sum(weights * ((league_points - hfa_term) - pts_allowed + opp_off))

            denom = np.sum(weights) + ridge_lambda
            new_off[team] = float(off_numer / denom)
            new_def[team] = float(def_numer / denom)

        # Center ratings to keep league mean consistent
        avg_off = float(np.mean(list(new_off.values())))
        avg_def = float(np.mean(list(new_def.values())))
        off_ratings = {k: v - avg_off for k, v in new_off.items()}
        def_ratings = {k: v - avg_def for k, v in new_def.items()}

    # Apply Bayesian prior shrinkage for small sample sizes
    final_power: dict[str | int, float] = {}
    final_off: dict[str | int, float] = {}
    final_def: dict[str | int, float] = {}
    for team in all_teams:
        n = stat_profiles.get(team, {}).get("games_played", 0)
        p = team_priors.get(team, 0.0)
        denom = n + prior_weight
        final_off[team] = float((n * off_ratings[team] + (prior_weight * p / 2.0)) / denom)
        final_def[team] = float((n * def_ratings[team] + (prior_weight * p / 2.0)) / denom)
        final_power[team] = float(final_off[team] + final_def[team])

    # Calculate Strength of Schedule (SOS)
    sos_dict: dict[str | int, float] = {}
    for team in all_teams:
        rows = merged[merged["team_id"] == team]
        if len(rows) > 0:
            opp_ratings = [final_power.get(opp, 0.0) for opp in rows["opponent_id"]]
            sos_dict[team] = float(np.mean(opp_ratings))
        else:
            sos_dict[team] = 0.0

    # Calculate Recent Form Rating (higher decay on recent 3-4 games)
    recent_form: dict[str | int, float] = {}
    for team in all_teams:
        rows = merged[merged["team_id"] == team]
        if len(rows) > 0:
            recent_weights = rows["recent_form_weight"].to_numpy()
            margin = (rows["points"] - rows["points_allowed"]).to_numpy()
            hfa_offset = np.where(
                rows["neutral_site"].astype(bool),
                0.0,
                np.where(rows["is_home"].astype(bool), home_advantage, -home_advantage),
            )
            adj_margin = margin - hfa_offset
            opp_ratings = np.array([final_power.get(opp, 0.0) for opp in rows["opponent_id"]])
            form_numer = np.sum(recent_weights * (adj_margin + opp_ratings))
            form_denom = np.sum(recent_weights)
            recent_form[team] = (
                float(form_numer / form_denom) if form_denom > 0 else final_power[team]
            )
        else:
            recent_form[team] = final_power[team]

    # Sort teams by power rating and build profiles
    sorted_teams = sorted(all_teams, key=lambda t: final_power[t], reverse=True)
    profiles: dict[str | int, TeamPowerProfile] = {}

    for rank, team in enumerate(sorted_teams, start=1):
        pr = final_power[team]
        stats_dict = stat_profiles.get(team, {})
        team_name = team_names.get(team) if team_names else None

        profiles[team] = TeamPowerProfile(
            team_id=team,
            team_name=team_name,
            rank=rank,
            tier=_assign_tier(pr),
            power_rating=pr,
            offensive_rating=final_off[team],
            defensive_rating=final_def[team],
            strength_of_schedule=sos_dict[team],
            recent_form_rating=recent_form[team],
            prior_rating=team_priors[team],
            games_played=stats_dict.get("games_played", 0),
            wins=stats_dict.get("wins", 0),
            losses=stats_dict.get("losses", 0),
            ties=stats_dict.get("ties", 0),
            yards_per_play=stats_dict.get("yards_per_play", 0.0),
            yards_per_play_allowed=stats_dict.get("yards_per_play_allowed", 0.0),
            passing_yards_per_play=stats_dict.get("passing_yards_per_play", 0.0),
            rushing_yards_per_play=stats_dict.get("rushing_yards_per_play", 0.0),
            third_down_rate=stats_dict.get("third_down_rate", 0.0),
            third_down_rate_allowed=stats_dict.get("third_down_rate_allowed", 0.0),
            red_zone_td_rate=stats_dict.get("red_zone_td_rate", 0.0),
            red_zone_td_rate_allowed=stats_dict.get("red_zone_td_rate_allowed", 0.0),
            turnovers_pg=stats_dict.get("turnovers_pg", 0.0),
            takeaways_pg=stats_dict.get("takeaways_pg", 0.0),
            turnover_margin_pg=stats_dict.get("turnover_margin_pg", 0.0),
            penalties_pg=stats_dict.get("penalties_pg", 0.0),
            penalty_yards_pg=stats_dict.get("penalty_yards_pg", 0.0),
            time_of_possession_sec=stats_dict.get("time_of_possession_sec", 0.0),
        )

    return PowerRankings(
        season=season,
        teams=profiles,
        home_advantage=home_advantage,
        league_points=league_points,
        margin_std=margin_std,
    )
