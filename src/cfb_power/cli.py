from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console

from .config import Paths
from .evaluation import evaluate_predictions
from .exports import export_all
from .features import build_matchup_features, build_team_pregame_features
from .model import fit_score_models, predict_matchups

app = typer.Typer(add_completion=False, help="CFB Power V1 forecast pipeline")
console = Console()


def _load_source(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not paths.games_csv.exists() or not paths.team_game_stats_csv.exists():
        raise typer.BadParameter("Missing normalized source data. Run create-demo-data or add CSV files.")
    return pd.read_csv(paths.games_csv), pd.read_csv(paths.team_game_stats_csv)


@app.command()
def create_demo_data(seasons: int = 3, teams: int = 12, weeks: int = 10) -> None:
    """Create synthetic data only to verify the V1 pipeline and workbook exports."""
    paths = Paths()
    paths.ensure()
    rng = np.random.default_rng(42)
    team_ids = [f"TEAM_{index:02d}" for index in range(1, teams + 1)]
    strength = {team: rng.normal(0, 5) for team in team_ids}
    game_rows: list[dict] = []
    stat_rows: list[dict] = []
    game_number = 0
    for season in range(2023, 2023 + seasons):
        for week in range(1, weeks + 1):
            order = list(rng.permutation(team_ids))
            for index in range(0, len(order) - 1, 2):
                away, home = order[index], order[index + 1]
                game_number += 1
                game_id = f"{season}-{week}-{game_number}"
                home_score = max(0, int(round(27 + strength[home] - 0.35 * strength[away] + 2.5 + rng.normal(0, 8))))
                away_score = max(0, int(round(25 + strength[away] - 0.35 * strength[home] + rng.normal(0, 8))))
                game_rows.append({
                    "game_id": game_id, "season": season, "week": week,
                    "game_date": f"{season}-{min(12, week):02d}-{min(28, 1 + week * 2):02d}T18:00:00Z",
                    "home_team_id": home, "away_team_id": away, "home_score": home_score,
                    "away_score": away_score, "neutral_site": False, "completed": True,
                })
                for team, opponent, is_home, points, allowed in [
                    (home, away, True, home_score, away_score),
                    (away, home, False, away_score, home_score),
                ]:
                    plays = int(rng.integers(55, 82))
                    total_yards = max(150, int(points * 12 + strength[team] * 8 + rng.normal(0, 55)))
                    stat_rows.append({
                        "game_id": game_id, "team_id": team, "opponent_id": opponent, "is_home": is_home,
                        "points": points, "points_allowed": allowed, "total_yards": total_yards,
                        "offensive_plays": plays, "passing_yards": int(total_yards * rng.uniform(0.45, 0.72)),
                        "rushing_yards": int(total_yards * rng.uniform(0.28, 0.55)),
                        "turnovers": int(rng.integers(0, 4)), "takeaways": int(rng.integers(0, 4)),
                        "third_down_attempts": int(rng.integers(9, 18)),
                        "third_down_conversions": int(rng.integers(3, 10)),
                        "red_zone_attempts": int(rng.integers(1, 6)),
                        "red_zone_touchdowns": int(rng.integers(0, 5)),
                        "penalties": int(rng.integers(2, 11)), "penalty_yards": int(rng.integers(15, 105)),
                        "time_of_possession_seconds": int(rng.integers(1350, 2251)),
                    })
    pd.DataFrame(game_rows).to_csv(paths.games_csv, index=False)
    pd.DataFrame(stat_rows).to_csv(paths.team_game_stats_csv, index=False)
    console.print(f"[green]Created synthetic source data:[/] {paths.processed_dir}")


@app.command()
def train() -> None:
    """Build pregame features, fit V1 models, and export completed-game predictions."""
    paths = Paths()
    games, stats = _load_source(paths)
    team_features = build_team_pregame_features(games, stats)
    matchups = build_matchup_features(games, team_features)
    training = matchups[matchups["completed"].astype(bool)].copy()
    models = fit_score_models(training, paths)
    predictions = predict_matchups(matchups, models)
    predictions.to_parquet(paths.processed_dir / "predictions.parquet", index=False)
    team_features.to_parquet(paths.processed_dir / "team_pregame_features.parquet", index=False)
    console.print(f"[green]Trained {len(training)} completed matchups and saved model:[/] {paths.model_file}")


@app.command()
def export_workbook() -> None:
    """Write CSV and formatted XLSX exports from the latest predictions."""
    paths = Paths()
    prediction_path = paths.processed_dir / "predictions.parquet"
    feature_path = paths.processed_dir / "team_pregame_features.parquet"
    if not prediction_path.exists() or not feature_path.exists():
        raise typer.BadParameter("Run train before export-workbook.")
    predictions = pd.read_parquet(prediction_path)
    team_stats = pd.read_parquet(feature_path)
    performance = evaluate_predictions(predictions)
    historical_results = predictions[predictions["completed"].astype(bool)].copy()
    export_all(predictions, team_stats, historical_results, performance, paths)
    console.print(f"[green]Wrote Excel workbook:[/] {paths.workbook_xlsx}")


if __name__ == "__main__":
    app()
