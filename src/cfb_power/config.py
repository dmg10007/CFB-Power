from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_dir(root: Path, env_var: str, default: str) -> Path:
    val = os.getenv(env_var)
    if val:
        path = Path(val)
        return path if path.is_absolute() else root / path
    return root / default


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    data_dir: Path | None = None
    export_dir: Path | None = None
    model_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.data_dir is None:
            object.__setattr__(self, "data_dir", _resolve_dir(self.root, "CFB_DATA_DIR", "data"))
        elif not self.data_dir.is_absolute():
            object.__setattr__(self, "data_dir", self.root / self.data_dir)

        if self.export_dir is None:
            object.__setattr__(
                self, "export_dir", _resolve_dir(self.root, "CFB_EXPORT_DIR", "data/exports")
            )
        elif not self.export_dir.is_absolute():
            object.__setattr__(self, "export_dir", self.root / self.export_dir)

        if self.model_dir is None:
            object.__setattr__(
                self, "model_dir", _resolve_dir(self.root, "CFB_MODEL_DIR", "models")
            )
        elif not self.model_dir.is_absolute():
            object.__setattr__(self, "model_dir", self.root / self.model_dir)

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
