from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .config import Paths


def _excel_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy a frame and remove timezone metadata Excel cannot serialize."""
    safe = frame.copy()
    for column in safe.columns:
        if isinstance(safe[column].dtype, pd.DatetimeTZDtype):
            safe[column] = safe[column].dt.tz_localize(None)
    return safe


def _write_dataframe(ws, frame: pd.DataFrame) -> None:
    safe_frame = _excel_safe_frame(frame)
    ws.append(list(safe_frame.columns))
    for row in safe_frame.itertuples(index=False, name=None):
        ws.append(list(row))
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    header_fill = PatternFill("solid", fgColor="123047")
    for cell in ws[1]:
        cell.font = Font(color="FFFFFF", bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
    for index, column in enumerate(safe_frame.columns, start=1):
        values = [str(column), *[str(value) for value in safe_frame[column].head(100).fillna("")]]
        ws.column_dimensions[get_column_letter(index)].width = min(
            max(len(value) for value in values) + 2, 28
        )


def export_all(
    predictions: pd.DataFrame,
    team_stats: pd.DataFrame,
    historical_results: pd.DataFrame,
    performance: pd.DataFrame,
    paths: Paths,
) -> None:
    paths.ensure()
    predictions.to_csv(paths.predictions_csv, index=False)
    team_stats.to_csv(paths.team_stats_csv, index=False)
    historical_results.to_csv(paths.results_csv, index=False)
    performance.to_csv(paths.performance_csv, index=False)

    wb = Workbook()
    wb.remove(wb.active)
    readme = wb.create_sheet("README")
    readme.append(["CFB Power V1 Export"])
    readme.append(["Generated UTC", datetime.now(timezone.utc).isoformat()])
    readme.append(["Purpose", "Excel-ready output package; connect dashboard tables to these sheets or CSVs."])
    readme.append(["Data rule", "All features are derived from games before the matchup date."])
    readme.append(["Model", "Two Ridge expected-score models with residual-based score intervals."])
    readme.append(["Dashboard inputs", "Predictions, Team_Stats, Historical_Results, Model_Performance"])
    readme["A1"].font = Font(size=16, bold=True, color="FFFFFF")
    readme["A1"].fill = PatternFill("solid", fgColor="123047")
    readme.column_dimensions["A"].width = 22
    readme.column_dimensions["B"].width = 100

    for sheet_name, frame in {
        "Predictions": predictions,
        "Team_Stats": team_stats,
        "Historical_Results": historical_results,
        "Model_Performance": performance,
    }.items():
        ws = wb.create_sheet(sheet_name)
        _write_dataframe(ws, frame)
        if sheet_name == "Predictions" and "home_win_probability" in frame.columns and len(frame):
            probability_col = get_column_letter(frame.columns.get_loc("home_win_probability") + 1)
            ws.conditional_formatting.add(
                f"{probability_col}2:{probability_col}{len(frame) + 1}",
                ColorScaleRule(
                    start_type="min",
                    start_color="F8696B",
                    mid_type="percentile",
                    mid_value=50,
                    mid_color="FFEB84",
                    end_type="max",
                    end_color="63BE7B",
                ),
            )
    wb.save(paths.workbook_xlsx)
