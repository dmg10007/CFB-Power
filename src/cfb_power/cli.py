from __future__ import annotations

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import Paths
from .evaluation import evaluate_predictions
from .exports import export_all
from .features import build_matchup_features, build_team_pregame_features
from .model import fit_score_models, predict_matchups
from .power_ranking import fit_power_rankings

app = typer.Typer(add_completion=False, help="CFB Power V1 forecast pipeline")
console = Console()


def _load_source(pipeline_paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load normalized games and team game statistics CSVs."""
    if not pipeline_paths.games_csv.exists() or not pipeline_paths.team_game_stats_csv.exists():
        raise typer.BadParameter(
            "Missing normalized source data. Run create-demo-data or add CSV files."
        )
    return pd.read_csv(pipeline_paths.games_csv), pd.read_csv(pipeline_paths.team_game_stats_csv)


@app.command()
def create_demo_data(
    seasons: int = typer.Option(3, min=1, help="Number of seasons to generate (minimum 1)"),
    teams: int = typer.Option(12, min=2, help="Number of teams to generate (minimum 2)"),
    weeks: int = typer.Option(10, min=1, help="Number of regular season weeks (minimum 1)"),
) -> None:
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
                home_score = max(
                    0,
                    int(
                        round(27 + strength[home] - 0.35 * strength[away] + 2.5 + rng.normal(0, 8))
                    ),
                )
                away_score = max(
                    0, int(round(25 + strength[away] - 0.35 * strength[home] + rng.normal(0, 8)))
                )
                game_rows.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "week": week,
                        "game_date": f"{season}-{min(12, week):02d}-{min(28, 1 + week * 2):02d}T18:00:00Z",
                        "home_team_id": home,
                        "away_team_id": away,
                        "home_score": home_score,
                        "away_score": away_score,
                        "neutral_site": False,
                        "completed": True,
                    }
                )
                for team, opponent, is_home, points, allowed in [
                    (home, away, True, home_score, away_score),
                    (away, home, False, away_score, home_score),
                ]:
                    plays = int(rng.integers(55, 82))
                    total_yards = max(
                        150, int(points * 12 + strength[team] * 8 + rng.normal(0, 55))
                    )
                    stat_rows.append(
                        {
                            "game_id": game_id,
                            "team_id": team,
                            "opponent_id": opponent,
                            "is_home": is_home,
                            "points": points,
                            "points_allowed": allowed,
                            "total_yards": total_yards,
                            "offensive_plays": plays,
                            "passing_yards": int(total_yards * rng.uniform(0.45, 0.72)),
                            "rushing_yards": int(total_yards * rng.uniform(0.28, 0.55)),
                            "turnovers": int(rng.integers(0, 4)),
                            "takeaways": int(rng.integers(0, 4)),
                            "third_down_attempts": int(rng.integers(9, 18)),
                            "third_down_conversions": int(rng.integers(3, 10)),
                            "red_zone_attempts": int(rng.integers(1, 6)),
                            "red_zone_touchdowns": int(rng.integers(0, 5)),
                            "penalties": int(rng.integers(2, 11)),
                            "penalty_yards": int(rng.integers(15, 105)),
                            "time_of_possession_seconds": int(rng.integers(1350, 2251)),
                        }
                    )
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
    console.print(
        f"[green]Trained {len(training)} completed matchups and saved model:[/] {paths.model_file}"
    )


@app.command()
def export_workbook() -> None:
    """Write CSV and formatted XLSX exports from the latest predictions and power rankings."""
    paths = Paths()
    prediction_path = paths.processed_dir / "predictions.parquet"
    feature_path = paths.processed_dir / "team_pregame_features.parquet"
    if not prediction_path.exists() or not feature_path.exists():
        raise typer.BadParameter("Run train before export-workbook.")
    predictions = pd.read_parquet(prediction_path)
    team_stats = pd.read_parquet(feature_path)
    performance = evaluate_predictions(predictions)
    historical_results = predictions[predictions["completed"].astype(bool)].copy()

    # Compute power rankings if source tables are available
    rankings_df = None
    if paths.games_csv.exists() and paths.team_game_stats_csv.exists():
        try:
            games, stats = _load_source(paths)
            rankings = fit_power_rankings(games, stats)
            rankings_df = rankings.to_dataframe()
        except Exception:
            rankings_df = None

    export_all(
        predictions, team_stats, historical_results, performance, paths, power_rankings=rankings_df
    )
    console.print(f"[green]Wrote Excel workbook:[/] {paths.workbook_xlsx}")


@app.command()
def power_rankings(
    top: int = typer.Option(25, min=1, help="Number of teams to display"),
    season: int | None = typer.Option(None, help="Filter to specific season"),
) -> None:
    """Compute and display opponent-adjusted CFB Power Rankings."""
    paths = Paths()
    games, stats = _load_source(paths)
    if season is not None:
        games = games[games["season"] == season]
    rankings = fit_power_rankings(games, stats)
    df = rankings.to_dataframe()

    table = Table(
        title=f"CFB Power Rankings ({rankings.season})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Rank", justify="right", style="bold")
    table.add_column("Team", style="bold white")
    table.add_column("Rating", justify="right", style="green")
    table.add_column("Tier", style="yellow")
    table.add_column("Record", justify="center")
    table.add_column("Offense", justify="right")
    table.add_column("Defense", justify="right")
    table.add_column("SOS", justify="right")
    table.add_column("Recent", justify="right")
    table.add_column("YPP", justify="right")
    table.add_column("3rd Down", justify="right")
    table.add_column("TO Marg", justify="right")

    for _, row in df.head(top).iterrows():
        table.add_row(
            str(row["rank"]),
            str(row["team_name"]),
            f"{row['power_rating']:+.1f}",
            str(row["tier"]).replace("Tier ", "T"),
            str(row["record"]),
            f"{row['offensive_rating']:+.1f}",
            f"{row['defensive_rating']:+.1f}",
            f"{row['strength_of_schedule']:+.1f}",
            f"{row['recent_form_rating']:+.1f}",
            f"{row['yards_per_play']:.2f}",
            f"{row['third_down_rate']:.1%}",
            f"{row['turnover_margin_pg']:+.1f}",
        )
    console.print(table)


@app.command()
def compare(
    team_a: str = typer.Argument(..., help="Team A ID or display name"),
    team_b: str = typer.Argument(..., help="Team B ID or display name"),
    neutral: bool = typer.Option(False, "--neutral", "-n", help="Played at neutral site"),
    home_team: str | None = typer.Option(
        None,
        "--home-team",
        "-h",
        help="Explicit home team (defaults to Team A if not neutral)",
    ),
) -> None:
    """Compare any two teams head-to-head and project score, spread, and statistical edges."""
    paths = Paths()
    games, stats = _load_source(paths)
    rankings = fit_power_rankings(games, stats)

    try:
        comparison = rankings.compare_teams(
            team_a_id=team_a,
            team_b_id=team_b,
            neutral_site=neutral,
            home_team=home_team,
        )
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc

    site_str = (
        "Neutral Field"
        if comparison.neutral_site
        else f"Home: {comparison.home_team or comparison.team_a.display_name}"
    )
    forecast_text = (
        f"[bold cyan]{comparison.team_a.display_name}[/] vs [bold cyan]{comparison.team_b.display_name}[/]\n"
        f"Site: [bold]{site_str}[/]\n"
        f"Projected Score: [bold green]{comparison.team_a.display_name} {comparison.team_a_expected_points:.1f}[/] - "
        f"[bold green]{comparison.team_b.display_name} {comparison.team_b_expected_points:.1f}[/]\n"
        f"Projected Spread: [bold yellow]{comparison.favored_team}[/] | Projected Total: [bold]{comparison.projected_total:.1f}[/]\n"
        f"Win Probability: [bold]{comparison.team_a.display_name} {comparison.team_a_win_probability:.1%}[/] | "
        f"[bold]{comparison.team_b.display_name} {comparison.team_b_win_probability:.1%}[/] (Confidence: {comparison.confidence_tier})"
    )
    console.print(Panel(forecast_text, title="Matchup Forecast", expand=False))

    if comparison.edge_summary:
        console.print("[bold yellow]Key Matchup Edges:[/]")
        for edge in comparison.edge_summary:
            console.print(f" • {edge}")

    table = Table(title="Tale of the Tape", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="bold")
    table.add_column(comparison.team_a.display_name, justify="center")
    table.add_column(comparison.team_b.display_name, justify="center")
    for _, row in comparison.stat_comparison.iterrows():
        table.add_row(
            str(row["Metric"]),
            str(row[comparison.team_a.display_name]),
            str(row[comparison.team_b.display_name]),
        )
    console.print(table)


if __name__ == "__main__":
    app()
