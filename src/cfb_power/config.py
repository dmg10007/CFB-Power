from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / os.getenv("CFB_DATA_DIR", "data")
    export_dir: Path = PROJECT_ROOT / os.getenv("CFB_EXPORT_DIR", "data/exports")
    model_dir: Path = PROJECT_ROOT / os.getenv("CFB_MODEL_DIR", "models")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def games_csv(self) -> Path:
        return self.processed_dir / "games.csv"

    @property
    def team_game_stats_csv(self) -> Path:
        return self.processed_dir / "team_game_stats.csv"

    @property
    def predictions_csv(self) -> Path:
        return self.export_dir / "predictions.csv"

    @property
    def team_stats_csv(self) -> Path:
        return self.export_dir / "team_stats.csv"

    @property
    def results_csv(self) -> Path:
        return self.export_dir / "historical_results.csv"

    @property
    def performance_csv(self) -> Path:
        return self.export_dir / "model_performance.csv"

    @property
    def workbook_xlsx(self) -> Path:
        return self.export_dir / "cfb_forecast_export.xlsx"

    @property
    def model_file(self) -> Path:
        return self.model_dir / "score_models.joblib"

    def ensure(self) -> None:
        for directory in (
            self.raw_dir,
            self.processed_dir,
            self.export_dir,
            self.model_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
