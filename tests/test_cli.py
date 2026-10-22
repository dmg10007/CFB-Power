"""Tests for the CFB-Power Typer CLI interface."""

from __future__ import annotations

from typer.testing import CliRunner

from cfb_power.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "create-demo-data" in result.output
    assert "train" in result.output
    assert "export-workbook" in result.output


def test_create_demo_data_validation():
    # seasons < 1 should fail
    result = runner.invoke(app, ["create-demo-data", "--seasons", "0"])
    assert result.exit_code != 0

    # teams < 2 should fail
    result = runner.invoke(app, ["create-demo-data", "--teams", "1"])
    assert result.exit_code != 0

    # weeks < 1 should fail
    result = runner.invoke(app, ["create-demo-data", "--weeks", "0"])
    assert result.exit_code != 0


def test_train_missing_data_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("CFB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CFB_EXPORT_DIR", str(tmp_path / "exports"))
    monkeypatch.setenv("CFB_MODEL_DIR", str(tmp_path / "models"))

    result = runner.invoke(app, ["train"])
    assert result.exit_code != 0
    assert "Missing normalized source data" in result.output


def test_export_workbook_before_training_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("CFB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CFB_EXPORT_DIR", str(tmp_path / "exports"))
    monkeypatch.setenv("CFB_MODEL_DIR", str(tmp_path / "models"))

    result = runner.invoke(app, ["export-workbook"])
    assert result.exit_code != 0
    assert "Run train before export-workbook" in result.output


def test_full_pipeline_cli_integration(tmp_path, monkeypatch):
    monkeypatch.setenv("CFB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CFB_EXPORT_DIR", str(tmp_path / "exports"))
    monkeypatch.setenv("CFB_MODEL_DIR", str(tmp_path / "models"))

    # 1. Create synthetic data with enough games (e.g. 3 seasons, 6 teams, 8 weeks = 72 games)
    create_result = runner.invoke(
        app,
        ["create-demo-data", "--seasons", "3", "--teams", "6", "--weeks", "8"],
    )
    assert create_result.exit_code == 0, create_result.output
    data_dir = tmp_path / "data" / "processed"
    assert (data_dir / "games.csv").exists()
    assert (data_dir / "team_game_stats.csv").exists()

    # 2. Train model
    train_result = runner.invoke(app, ["train"])
    assert train_result.exit_code == 0, train_result.output
    assert (data_dir / "predictions.parquet").exists()
    assert (tmp_path / "models" / "score_models.joblib").exists()

    # 3. Export workbook
    export_result = runner.invoke(app, ["export-workbook"])
    assert export_result.exit_code == 0, export_result.output
    export_dir = tmp_path / "exports"
    assert (export_dir / "predictions.csv").exists()
    assert (export_dir / "team_stats.csv").exists()
    assert (export_dir / "historical_results.csv").exists()
    assert (export_dir / "model_performance.csv").exists()
    assert (export_dir / "cfb_forecast_export.xlsx").exists()
