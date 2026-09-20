import pandas as pd
from openpyxl import load_workbook

from cfb_power.config import Paths
from cfb_power.exports import export_all


def test_export_all_writes_timezone_aware_dates_to_excel(tmp_path):
    paths = Paths(
        root=tmp_path,
        data_dir=tmp_path / "data",
        export_dir=tmp_path / "data" / "exports",
        model_dir=tmp_path / "models",
    )
    predictions = pd.DataFrame({
        "game_id": ["g1"],
        "game_date": pd.to_datetime(["2025-09-01T18:00:00Z"], utc=True),
        "home_win_probability": [0.6],
    })
    team_stats = pd.DataFrame({
        "game_id": ["g1"],
        "game_date": pd.to_datetime(["2025-09-01T18:00:00Z"], utc=True),
    })
    export_all(predictions, team_stats, predictions.copy(), pd.DataFrame(), paths)
    assert paths.workbook_xlsx.exists()
    workbook = load_workbook(paths.workbook_xlsx, read_only=True)
    assert workbook.sheetnames == [
        "README",
        "Predictions",
        "Team_Stats",
        "Historical_Results",
        "Model_Performance",
    ]
