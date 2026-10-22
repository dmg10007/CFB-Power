from __future__ import annotations

import json
import shutil
from pathlib import Path

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
from .upcoming import _team_names

app = typer.Typer(add_completion=False, help="CFB Power V1 forecast pipeline")
console = Console()


def _find_dataset_names(dataset_dir: Path) -> dict[str | int, str]:
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return _team_names(manifest, [])
        except Exception:
            pass
    return {}


def _resolve_source_data(
    pipeline_paths: Paths, dataset: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str | int, str]]:
    """Resolve games, team stats, and team display names from dataset or processed dir."""
    if dataset:
        target = Path(dataset)
        if not target.is_dir():
            target = pipeline_paths.data_dir / "datasets" / dataset
        if not (target / "games.csv").exists() or not (target / "team_game_stats.csv").exists():
            raise typer.BadParameter(f"Dataset directory not found: {dataset}")
        games = pd.read_csv(target / "games.csv")
        stats = pd.read_csv(target / "team_game_stats.csv")
        names = _find_dataset_names(target)
        return games, stats, names

    if pipeline_paths.games_csv.exists() and pipeline_paths.team_game_stats_csv.exists():
        games = pd.read_csv(pipeline_paths.games_csv)
        stats = pd.read_csv(pipeline_paths.team_game_stats_csv)
        names: dict[str | int, str] = {}
        datasets_dir = pipeline_paths.data_dir / "datasets"
        if datasets_dir.exists():
            candidates = sorted(
                datasets_dir.glob("*/manifest.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                names = _find_dataset_names(candidates[0].parent)
        return games, stats, names

    # Fallback to latest dataset if active processed dir is unpopulated
    datasets_dir = pipeline_paths.data_dir / "datasets"
    if datasets_dir.exists():
        candidates = sorted(
            [d for d in datasets_dir.iterdir() if d.is_dir() and (d / "games.csv").exists()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            latest = candidates[0]
            games = pd.read_csv(latest / "games.csv")
            stats = pd.read_csv(latest / "team_game_stats.csv")
            names = _find_dataset_names(latest)
            return games, stats, names

    raise typer.BadParameter(
        "Missing normalized source data. Run create-demo-data or capture a CFBD dataset."
    )


def _load_source(pipeline_paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load normalized games and team game statistics CSVs."""
    games, stats, _ = _resolve_source_data(pipeline_paths)
    return games, stats


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
    names: dict[str | int, str] = {}
    if paths.games_csv.exists() and paths.team_game_stats_csv.exists():
        try:
            games, stats, names = _resolve_source_data(paths)
            rankings = fit_power_rankings(games, stats, team_names=names)
            rankings_df = rankings.to_dataframe()
        except Exception:
            rankings_df = None

    export_all(
        predictions,
        team_stats,
        historical_results,
        performance,
        paths,
        power_rankings=rankings_df,
        team_names=names,
    )
    console.print(f"[green]Wrote Excel workbook:[/] {paths.workbook_xlsx}")


@app.command()
def activate_dataset(
    dataset: str = typer.Argument(
        None,
        help="Dataset ID or directory to activate into data/processed/. If omitted, uses latest verified dataset.",
    ),
) -> None:
    """Copy an assembled dataset into active data/processed/ for pipeline training."""
    paths = Paths()
    paths.ensure()
    datasets_dir = paths.data_dir / "datasets"

    if dataset:
        target = Path(dataset)
        if not target.is_dir():
            target = datasets_dir / dataset
    else:
        if not datasets_dir.exists():
            raise typer.BadParameter("No datasets directory found.")
        candidates = sorted(
            [d for d in datasets_dir.iterdir() if d.is_dir() and (d / "games.csv").exists()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise typer.BadParameter("No verified datasets found to activate.")
        target = candidates[0]

    if not (target / "games.csv").exists() or not (target / "team_game_stats.csv").exists():
        raise typer.BadParameter(f"Dataset incomplete or not found: {target}")

    shutil.copy2(target / "games.csv", paths.games_csv)
    shutil.copy2(target / "team_game_stats.csv", paths.team_game_stats_csv)
    console.print(f"[green]Activated dataset into processed directory:[/] {target.name}")


@app.command()
def power_rankings(
    top: int = typer.Option(25, min=1, help="Number of teams to display"),
    season: int | None = typer.Option(None, help="Filter to specific season"),
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Optional dataset ID or path (defaults to active data or latest dataset)",
    ),
) -> None:
    """Compute and display opponent-adjusted CFB Power Rankings."""
    paths = Paths()
    games, stats, names = _resolve_source_data(paths, dataset=dataset)
    if season is not None:
        games = games[games["season"] == season]
    rankings = fit_power_rankings(games, stats, team_names=names)
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
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Optional dataset ID or path (defaults to active data or latest dataset)",
    ),
) -> None:
    """Compare any two teams head-to-head and project score, spread, and statistical edges."""
    paths = Paths()
    games, stats, names = _resolve_source_data(paths, dataset=dataset)
    rankings = fit_power_rankings(games, stats, team_names=names)

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


@app.command(name="sunday-update")
@app.command(name="weekly-update")
def sunday_update(
    season: int = typer.Option(2026, "--season", "-s", help="Season year"),
    completed_week: int = typer.Option(
        ..., "--completed-week", "-c", help="Week that just finished"
    ),
    upcoming_week: int = typer.Option(
        ..., "--upcoming-week", "-u", help="Upcoming week to forecast"
    ),
    api_key: str | None = typer.Option(
        None, "--api-key", "-k", envvar="CFBD_API_KEY", help="CFBD API key"
    ),
    min_history: int = typer.Option(3, "--min-history", help="Minimum prior games threshold"),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Run offline using existing downloaded dataset without CFBD API calls",
    ),
) -> None:
    """Execute Sunday model evaluation, adaptive calibration, and next week's power ranking forecasts."""
    from .weekly_update import run_sunday_update

    paths = Paths()
    console.print(f"[bold green]Starting Sunday update cycle for Season {season}...[/]")
    result = run_sunday_update(
        data_dir=paths.data_dir,
        season=season,
        completed_week=completed_week,
        upcoming_week=upcoming_week,
        api_key=api_key,
        emit=lambda msg: console.print(f"  [cyan]•[/] {msg}"),
        min_history_games=min_history,
        offline=offline,
    )

    # Display evaluation summary
    if result.evaluation:
        ev = result.evaluation
        eval_text = (
            f"Games Evaluated: [bold]{ev.get('scored', 0)}[/] of {ev.get('games_in_scope', 0)}\n"
            f"Winner Accuracy: [bold green]{ev.get('winner_accuracy', 0.0):.1%}[/]\n"
            f"Margin MAE: [bold yellow]{ev.get('margin_mae', 0.0):.1f} pts[/] | Total MAE: [bold]{ev.get('total_mae', 0.0):.1f} pts[/]\n"
            f"Brier Calibration: [bold]{ev.get('brier_score', 0.0):.3f}[/] | Margin Bias: [bold]{ev.get('margin_bias', 0.0):+.1f} pts[/]"
        )
    else:
        eval_text = "No previous pregame forecasts found for evaluation; skipped score comparison."
    console.print(Panel(eval_text, title=f"Week {completed_week} Model Evaluation", expand=False))

    # Display calibration
    cal = result.calibration
    cal_text = (
        f"Calibrated Home-Field Advantage: [bold green]+{cal.get('calibrated_hfa', 2.5):.2f} pts[/] "
        f"(Actual Season Avg: {cal.get('actual_home_margin_avg', 2.5):+.2f})\n"
        f"Calibrated Margin Residual Std: [bold]{cal.get('margin_std', 13.5):.2f}[/]\n"
        f"Ranked FBS Teams: [bold]{result.rankings_count}[/] across all conferences"
    )
    console.print(Panel(cal_text, title="Adaptive Model Calibration", expand=False))

    # Top teams table
    table = Table(
        title="Updated Power Rankings (Top 10)", show_header=True, header_style="bold magenta"
    )
    table.add_column("Rank", justify="center")
    table.add_column("Team", style="bold")
    table.add_column("Rating", justify="right")
    table.add_column("Tier", justify="center")
    for t in result.top_teams:
        table.add_row(
            str(t["rank"]), str(t["team_name"]), f"{t['power_rating']:+.2f}", str(t["tier"])
        )
    console.print(table)

    console.print(f"[bold green]✓ Week {upcoming_week} forecasts ready:[/] {result.forecast_dir}")
    console.print(f"[bold green]✓ Updated Excel workbook:[/] {result.workbook_path}")


if __name__ == "__main__":
    app()
