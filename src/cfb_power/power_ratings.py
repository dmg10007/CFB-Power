"""Experimental, score-based opponent-adjusted power ratings (not production forecasts)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

MODEL_VERSION = "opponent_adjusted_scores_v1"
GAME_FIELDS = {"game_id", "season", "game_date", "home_team_id", "away_team_id",
               "home_score", "away_score", "neutral_site", "completed"}


def utc_timestamp(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("A nonmissing, timezone-aware timestamp is required")
    return stamp.tz_convert("UTC")


def checked_games(games: pd.DataFrame) -> pd.DataFrame:
    missing = GAME_FIELDS - set(games.columns)
    if missing:
        raise ValueError(f"Missing game fields: {sorted(missing)}")
    result = games.copy()
    if result.game_id.duplicated().any():
        raise ValueError("Duplicate game IDs")
    result["game_date"] = pd.to_datetime([utc_timestamp(x) for x in result.game_date], utc=True)
    for name in ("game_id", "season", "home_team_id", "away_team_id"):
        values = pd.to_numeric(result[name], errors="raise")
        if (~np.isfinite(values) | (values <= 0) | (values % 1 != 0)).any():
            raise ValueError(f"Invalid positive integer field: {name}")
        result[name] = values.astype("int64")
    for name in ("completed", "neutral_site"):
        if not result[name].isin([True, False]).all():
            raise ValueError(f"Invalid boolean field: {name}")
        result[name] = result[name].astype(bool)
    if (result.home_team_id == result.away_team_id).any():
        raise ValueError("A team cannot play itself")
    for name in ("home_score", "away_score"):
        values = pd.to_numeric(result[name], errors="raise")
        bad = ~np.isfinite(values) | (values < 0) | (values % 1 != 0)
        if (bad & result.completed).any():
            raise ValueError(f"Invalid completed score: {name}")
        result[name] = values
    if "result_available_at_utc" in result:
        present = result.result_available_at_utc.notna()
        result["result_available_at_utc"] = pd.to_datetime(
            [utc_timestamp(x) if pd.notna(x) else pd.NaT
             for x in result.result_available_at_utc], utc=True)
        if (present & (result.result_available_at_utc < result.game_date)).any():
            raise ValueError("Result availability precedes kickoff")
    return result.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def eligible_history(games: pd.DataFrame, as_of) -> pd.DataFrame:
    cutoff = utc_timestamp(as_of)
    mask = games.completed & (games.game_date < cutoff)
    if "result_available_at_utc" in games:
        mask &= games.result_available_at_utc.notna() & (games.result_available_at_utc < cutoff)
    return games.loc[mask].copy()


@dataclass(frozen=True)
class RatingSettings:
    half_life_days: float = 35.0
    team_penalty: float = 4.0
    venue_penalty: float = 10.0
    previous_season_carry: float = 0.5
    low_history_games: int = 3

    def __post_init__(self):
        for name in ("half_life_days", "team_penalty", "venue_penalty"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.previous_season_carry) or not 0 <= self.previous_season_carry <= 1:
            raise ValueError("previous_season_carry must be in [0, 1]")
        if type(self.low_history_games) is not int or self.low_history_games < 0:
            raise ValueError("low_history_games must be a nonnegative integer")


@dataclass
class RatingSnapshot:
    teams: pd.DataFrame
    base_points: float
    home_field_points: float
    metadata: dict

    def compare(self, team_a: int, team_b: int, venue: str = "neutral") -> dict:
        if team_a == team_b:
            raise ValueError("Choose two different teams")
        signs = {"neutral": 0, "team_a_home": 1, "team_b_home": -1}
        if venue not in signs:
            raise ValueError(f"venue must be one of {tuple(signs)}")
        table = self.teams.set_index("team_id")
        if team_a not in table.index or team_b not in table.index:
            raise ValueError("Both teams must belong to this rating snapshot")
        a, b = table.loc[team_a], table.loc[team_b]
        venue_margin = signs[venue] * self.home_field_points
        pa = self.base_points + a.offense_rating - b.defense_rating + venue_margin / 2
        pb = self.base_points + b.offense_rating - a.defense_rating - venue_margin / 2
        if min(pa, pb) < 0:
            raise ValueError("Negative linear score estimate; matchup is outside the supported scoring range")
        warnings = ["Experimental score estimates; win probabilities and intervals are not calibrated"]
        if a.low_history or b.low_history:
            warnings.append("At least one team has limited current-season history")
        if a.component_id != b.component_id:
            warnings.append("Teams are in disconnected current-season schedule components; priors anchor the comparison")
        return {"team_a_id": int(team_a), "team_b_id": int(team_b), "venue": venue,
                "team_a_expected_points": float(pa), "team_b_expected_points": float(pb),
                "margin_a": float(pa - pb), "total": float(pa + pb),
                "venue_margin": float(venue_margin), "warnings": warnings,
                "model_version": MODEL_VERSION, "as_of_utc": self.metadata["as_of_utc"]}


def _fit(games, ids, anchor, settings, prior=None):
    n = len(ids)
    positions = {team: i for i, team in enumerate(ids)}
    target_prior = np.zeros(2 * n + 2)
    if prior is not None:
        previous = prior.teams.set_index("team_id")
        for i, team in enumerate(ids):
            if team in previous.index:
                target_prior[1 + i] = settings.previous_season_carry * previous.loc[team, "offense_rating"]
                target_prior[1 + n + i] = settings.previous_season_carry * previous.loc[team, "defense_rating"]
        target_prior[1:1+n] -= target_prior[1:1+n].mean()
        target_prior[1+n:1+2*n] -= target_prior[1+n:1+2*n].mean()
    penalties = np.full(2 * n + 2, settings.team_penalty)
    penalties[0], penalties[-1] = 0, settings.venue_penalty
    x, y, weights = [], [], []
    for g in games.itertuples(index=False):
        weight = 2.0 ** (-max((anchor - g.game_date).total_seconds() / 86400, 0) / settings.half_life_days)
        for own, opponent, score, sign in (
            (g.home_team_id, g.away_team_id, g.home_score, 1),
            (g.away_team_id, g.home_team_id, g.away_score, -1)):
            row = np.zeros(2 * n + 2)
            row[0] = 1
            row[1 + positions[own]] = 1
            row[1 + n + positions[opponent]] = -1
            row[-1] = 0 if g.neutral_site else sign / 2
            x.append(row)
            y.append(score)
            weights.append(weight)
    x, y, weights = np.asarray(x), np.asarray(y), np.asarray(weights)
    if weights.sum() <= 1e-10:
        raise ValueError("Historical weights have negligible mass; use a later data capture or longer half-life")
    system = x.T @ (weights[:, None] * x) + np.diag(penalties)
    theta = np.linalg.solve(system, x.T @ (weights * y) + penalties * target_prior)
    offense, defense = theta[1:1+n].copy(), theta[1+n:1+2*n].copy()
    mean_o, mean_d = offense.mean(), defense.mean()
    offense -= mean_o
    defense -= mean_d
    base = theta[0] + mean_o - mean_d
    return offense, defense, float(base), float(theta[-1])


def _components(ids, games):
    parent = {team: team for team in ids}
    def find(team):
        while parent[team] != team:
            parent[team] = parent[parent[team]]
            team = parent[team]
        return team
    for g in games.itertuples(index=False):
        a, b = find(g.home_team_id), find(g.away_team_id)
        parent[max(a, b)] = min(a, b)
    return {team: find(team) for team in ids}


def build_power_ratings(games, *, season, as_of, settings=None, team_ids=None) -> RatingSnapshot:
    settings = settings or RatingSettings()
    if type(season) is not int or season <= 0:
        raise ValueError("season must be a positive integer")
    cutoff = utc_timestamp(as_of)
    checked = checked_games(games)
    history = eligible_history(checked, cutoff)
    current = history[history.season == season]
    previous = history[history.season == season - 1]
    observed = checked[(checked.season == season) & (checked.game_date < cutoff)]
    represented = set(observed.home_team_id) | set(observed.away_team_id)
    represented |= set(previous.home_team_id) | set(previous.away_team_id)
    if team_ids is not None:
        supplied = list(team_ids)
        if any(type(x) is not int or x <= 0 for x in supplied) or len(set(supplied)) != len(supplied):
            raise ValueError("team_ids must contain distinct positive integers")
        represented |= set(supplied)
    ids = sorted(int(x) for x in represented)
    if not ids or (current.empty and previous.empty):
        raise ValueError("No eligible current- or previous-season completed games")
    prior = None
    if not previous.empty:
        prior_ids = sorted(set(previous.home_team_id) | set(previous.away_team_id))
        o, d, base, hfa = _fit(previous, prior_ids, previous.game_date.max(), settings)
        prior = RatingSnapshot(pd.DataFrame({"team_id": prior_ids, "offense_rating": o,
                                            "defense_rating": d}), base, hfa, {})
    if current.empty:
        table = prior.teams.set_index("team_id").reindex(ids).fillna(0)
        o = settings.previous_season_carry * table.offense_rating.to_numpy()
        d = settings.previous_season_carry * table.defense_rating.to_numpy()
        o -= o.mean()
        d -= d.mean()
        base, hfa = prior.base_points, 0.0
    else:
        o, d, base, hfa = _fit(current, ids, cutoff, settings, prior)
    components = _components(ids, current)
    power = dict(zip(ids, o + d))
    rows = []
    for i, team in enumerate(ids):
        played = current[(current.home_team_id == team) | (current.away_team_id == team)]
        past = previous[(previous.home_team_id == team) | (previous.away_team_id == team)]
        weighted_opponents, effective = [], []
        for g in played.itertuples(index=False):
            opponent = g.away_team_id if g.home_team_id == team else g.home_team_id
            weight = 2 ** (-(cutoff - g.game_date).total_seconds() / 86400 / settings.half_life_days)
            weighted_opponents.append(power[opponent])
            effective.append(weight)
        rows.append({"team_id": team, "offense_rating": float(o[i]), "defense_rating": float(d[i]),
                     "power_rating": float(o[i] + d[i]), "games_played": len(played),
                     "effective_games": float(sum(effective)), "previous_season_games": len(past),
                     "prior_source": "previous_season" if len(past) else "league_average",
                     "schedule_strength": float(np.average(weighted_opponents, weights=effective)) if effective else np.nan,
                     "low_history": len(played) < settings.low_history_games,
                     "component_id": components[team]})
    teams = pd.DataFrame(rows).sort_values(["power_rating", "team_id"], ascending=[False, True]).reset_index(drop=True)
    teams.insert(0, "power_rank", teams.power_rating.rank(method="min", ascending=False).astype(int))
    warnings = ["Experimental score-only ratings; no richer statistics or player adjustments are active",
                "Team universe is dataset/explicit-ID coverage, not a verified complete FBS membership list",
                "Hyperparameters are fixed engineering defaults, not validated optimal settings",
                "No calibrated uncertainty intervals or win probabilities are provided"]
    if "result_available_at_utc" not in checked:
        warnings.append("Result knowledge times are absent: historical cutoff is kickoff-based, not proof of point-in-time availability")
    if len(set(components.values())) > 1:
        warnings.append("Disconnected current-season schedule: cross-component comparisons depend on priors and regularization")
    unresolved = int((~observed.completed).sum())
    if unresolved:
        warnings.append(f"{unresolved} past-kickoff games have no completed result and are excluded")
    if not current.empty and (cutoff - current.game_date.max()).total_seconds() > 9 * 86400:
        warnings.append("Current-season completed-game history is more than 9 days old")
    if current.empty:
        warnings.append("No current-season completed games: rankings use discounted previous-season ratings; venue effect is zero")
    return RatingSnapshot(teams, float(base), float(hfa), {
        "model_version": MODEL_VERSION, "experimental": True, "season": season,
        "as_of_utc": cutoff.isoformat(), "settings": asdict(settings),
        "current_games": len(current), "previous_games": len(previous),
        "training_game_ids": sorted(int(x) for x in pd.concat([current, previous]).game_id),
        "result_availability_filter_present": "result_available_at_utc" in checked,
        "warnings": warnings})
