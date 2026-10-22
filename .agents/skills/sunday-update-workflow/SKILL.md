---
name: sunday-update-workflow
description: >-
  Use this skill when the user asks to run the Sunday model evaluation,
  recalibration, weekly ingestion, or generate upcoming week forecasts.
---

# Sunday Model Update & Weekly Slate Runbook

This runbook guides the agent through the complete Sunday operational cycle:
1. Ingesting finalized results for the completed week and upcoming matchups for the next week.
2. Evaluating model predictions against actual final scores.
3. Recalibrating model parameters (in-season Home-Field Advantage and residual error scale).
4. Refitting opponent-adjusted Power Rankings with recency weighting.
5. Forecasting the upcoming week's slate and updating the Excel workbook.

---

## Quick Execution

### Option 1: Unified CLI Command (Recommended)
Run the Sunday update command with the completed week and the upcoming week:
```bash
cfb-power sunday-update --completed-week <COMPLETED_WEEK> --upcoming-week <UPCOMING_WEEK>
```
*Example for transitioning from Week 7 to Week 8:*
```bash
cfb-power sunday-update --completed-week 7 --upcoming-week 8
```

### Option 2: Offline Execution (If Quota or Network Issues Arise)
If CFBD rate limits or monthly quotas are hit, run in offline mode to use already-saved snapshots:
```bash
cfb-power sunday-update --completed-week 7 --upcoming-week 8 --offline
```

### Option 3: Streamlit Web Dashboard
1. Start the dashboard:
   ```bash
   streamlit run src/cfb_power/dashboard.py
   ```
2. Open the **`🔄 Sunday Weekly Update`** tab.
3. Select the **Season**, **Completed Week**, and **Upcoming Week**.
4. Click **🚀 Run Sunday Model Update & Forecast**.

---

## Detailed Step-by-Step Procedure

### Step 1: Pre-Flight Verification
1. Ensure the virtual environment is active:
   ```bash
   source .venv/bin/activate
   ```
2. Verify API key configuration in `.env` (`CFBD_API_KEY`).
3. Check available API quota if making live calls.

### Step 2: Run Ingestion and Model Update
Execute the Sunday cycle:
```bash
python -m cfb_power.cli sunday-update --completed-week <C> --upcoming-week <U>
```

The script automatically executes:
- Ingestion of completed games and team game stats for `<C>`.
- Ingestion of scheduled games for `<U>`.
- Assembly of verified snapshots and manifest generation.
- Evaluation of pregame predictions vs actual scores.
- Adaptive calibration of Home-Field Advantage (HFA) and margin standard deviation ($\sigma$).
- Exponential recency decay refit of Power Rankings.
- Slate generation with spreads, totals, win probabilities, and confidence intervals.
- Automatic export to `data/exports/cfb_forecast_export.xlsx`.

### Step 3: Verify Evaluation Metrics
Review the evaluation card output:
- **Winner Accuracy**: Target > 65% for established FBS matchups.
- **Margin MAE**: Target ~11.0 to 13.5 points.
- **Total MAE**: Target ~12.0 to 14.0 points.
- **Brier Score**: Target < 0.25 (well-calibrated probability distributions).
- **Margin Bias**: Check for systemic over/under-estimation of home teams.

### Step 4: Verify Forecast Deliverables
Ensure output files were generated and are non-empty:
1. **Latest Run Directory**:
   Check `data/runs/<dataset_id>/predictions_<timestamp>-<hash>/`
   - `predictions.csv`: Must contain all scheduled games with `projected_spread_home`, `projected_total`, `home_win_probability`, and `confidence_tier`.
   - `manifest.json`: Must list model parameters, input dataset SHA-256, and game counts.
2. **Master Excel Export**:
   Confirm [`data/exports/cfb_forecast_export.xlsx`](file:///home/dalton/CFB-Power/data/exports/cfb_forecast_export.xlsx) has been updated with:
   - `Power Rankings` sheet
   - `Weekly Forecasts` sheet
   - `Historical Results` sheet
   - `Model Metrics` sheet

---

## Troubleshooting Common Issues

### 1. HTTP 429 / CFBD Rate Limit
- **Symptom**: `CFBD request failed: HTTP 429 (Monthly call quota exceeded)`.
- **Solution**: Run with `--offline` flag to generate forecasts using already-ingested local data snapshots without making external calls.

### 2. Revision Conflict During Snapshot Assembly
- **Symptom**: `Snapshot revisions require review before assembly.`
- **Solution**: Open the dashboard at **`📥 Ingestion & Revisions`** tab, inspect conflicting game statistics, select the preferred snapshot, and click **Assemble Dataset**.

### 3. Missing Team Names in Output
- **Symptom**: Predictions output displays numeric team IDs instead of school names.
- **Solution**: Ensure team name resolution can find `games.json` in `data/raw/cfbd/` or run `python -m cfb_power.team_names`.
