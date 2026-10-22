# CFB Power

Team-stat-driven college football power rankings, situational matchup forecasting, and automated weekly operations pipeline with an interactive web dashboard and Excel-ready data packages.

---

## Overview

**CFB Power** is an end-to-end college football analytical engine designed to forecast matchup scores, spreads, totals, and win probabilities using only information available prior to kickoff.

The system combines:
1. **Opponent-Adjusted Power Rankings**: An iterative regularized Ridge regression solver determining offensive, defensive, and overall ratings, with strength-of-schedule (SOS) adjustments, recent form weighting, and situational efficiency metrics.
2. **Automated Weekly Operations Pipeline**: One-step Sunday update cycle that evaluates completed game forecasts, computes adaptive calibrations for Home-Field Advantage (HFA) and margin variance, updates power ratings, and publishes the upcoming slate's forecasts.
3. **Interactive Web Dashboard**: A 7-tab Streamlit interface for live ratings inspection, head-to-head matchup simulation ("Tale of the Tape"), forecast review, weekly update execution, results grading, and audited data ingestion.
4. **Immutable Audited Ingestion**: Direct CollegeFootballData (CFBD) API integration with rate-limiting, local raw audit caching with score verification, cryptographic SHA-256 manifests, and human-in-the-loop revision management.
5. **Digestible Exports**: Multi-sheet formatted Excel workbook (`cfb_forecast_export.xlsx`) and structured CSV outputs for external consumption and visualization.

---

## Key Features

- **Opponent-Adjusted Ratings**: Iterative Ridge regression solver pre-indexed and vectorized for high performance, decomposing team performance into baseline offensive and defensive strength per possession/play.
- **Conference Strength Weighting**: Resolves conference affiliations and incorporates empirical conference strength baselines into the Ridge prior shrinkage (`conference_weight=0.15`), appropriately weighting schedule difficulty across conferences.
- **Advanced Efficiency Metrics**: Ingests and normalizes down-by-down Success Rates, EPA / PPA per play, Explosiveness, and defensive Havoc Rates from CFBD advanced box statistics.
- **Betting Market Comparison & Value Detection**: Ingests consensus betting lines via CFBD `/lines`, computes model spread and total edges, and highlights value plays in the dashboard and workbook.
- **Situational Efficiency Analytics**: Tracks yards per play (YPP), 3rd-down conversion rates, red-zone scoring efficiency, and turnover margin per game.
- **Head-to-Head Matchup Forecaster**: Computes projected scores, point spreads, over/under totals, win probabilities, and confidence tiers, accompanied by automated situational edge summaries.
- **Adaptive Calibration**: Dynamically adjusts home-field advantage and margin residual standard deviation week-by-week based on observed empirical margins.
- **Strict No-Leakage Guarantee**: Rolling pregame windows (3-game, 5-game, season-to-date) shifted strictly before kickoff time.
- **Audited Reproducibility**: Raw API responses are archived immutably with SHA-256 manifests. Stale or conflicting score revisions require explicit human approval before dataset publication.
- **Multi-Device Dashboard**: Streamlit web dashboard accessible on desktop, mobile, and remote networks (e.g., via Tailscale).

---

## Architecture & Data Pipeline

```text
       Raw CFBD API Responses (with Local Audit Caching)
                             ↓
              Data Validation & Normalization
         (games.csv, team_game_stats.csv, manifests)
                             ↓
       Opponent-Adjusted Power Rankings & Situational Metrics
                             ↓
             Adaptive Calibration & Score Models
            (Dynamic HFA, Margin Residual Std Dev)
                             ↓
           Pregame Weekly Forecasts & Matchup Analysis
       (Projected Scores, Spreads, Totals, Win Probabilities)
                             ↓
        Streamlit Web Dashboard  |  Excel Workbook & CSV Exports
```

---

## Setup & Installation

### Requirements
- Python 3.12+
- Linux, macOS, or Windows WSL

### Installation

```bash
# Clone the repository
git clone https://github.com/dmg10007/CFB-Power.git
cd CFB-Power

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate         # Linux/macOS
# .venv\Scripts\Activate.ps1      # Windows PowerShell

# Install package in editable mode with development dependencies
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

### Environment Configuration

Copy `.env.example` to `.env` to configure API keys and paths:

```bash
cp .env.example .env
```

Key configuration options in `.env`:
- `CFBD_API_KEY`: Your CollegeFootballData API key (optional for demo data, required for live ingestion).
- `CFB_DATA_DIR`: Base directory for raw and processed datasets (default: `data/`).
- `CFB_EXPORT_DIR`: Output directory for CSV and Excel exports (default: `data/exports/`).
- `CFB_MODEL_DIR`: Directory where serialized models are stored (default: `models/`).

---

## Quick Start

### 1. Synthetic Demo Pipeline (No API Key Required)

Verify the full pipeline in seconds using synthetic data:

```bash
# Generate synthetic dataset
cfb-power create-demo-data --seasons 3 --teams 12 --weeks 10

# View opponent-adjusted Power Rankings
cfb-power power-rankings --top 10

# Compare two teams head-to-head
cfb-power compare TEAM_01 TEAM_02 --home-team TEAM_01

# Train score models and export Excel workbook
cfb-power train
cfb-power export-workbook
```

### 2. Launch the Web Dashboard

Start the interactive Streamlit dashboard:

```bash
streamlit run src/cfb_power/dashboard.py
```

Open `http://localhost:8501` in your browser.

#### Accessing over Tailscale or Local Network
To view the dashboard on your phone or remote devices via Tailscale:

```bash
streamlit run src/cfb_power/dashboard.py --server.address 0.0.0.0 --server.port 8501
```

> [!NOTE]
> Streamlit uses plain HTTP. When connecting from another device over Tailscale MagicDNS or IP, use `http://<your-tailscale-node>:8501` (avoid `https://`).

#### Native Desktop Launcher (Linux)
Manage the dashboard as a local background daemon:

```bash
python -m cfb_power.desktop start     # Launch daemon and open browser
python -m cfb_power.desktop status    # Check daemon health and port
python -m cfb_power.desktop stop      # Gracefully shut down daemon
```

---

## CLI Reference

The package provides the `cfb-power` CLI command:

```bash
# Display help and all commands
cfb-power --help

# 1. Power Rankings: View Top 25 opponent-adjusted team ratings
cfb-power power-rankings --top 25
cfb-power power-rankings --conference SEC --top 16
cfb-power power-rankings --season 2026 --top 50

# 2. Team Comparison: Head-to-head projection & situational edges
cfb-power compare "Georgia" "Alabama" --neutral
cfb-power compare "Ohio State" "Michigan" --home-team "Ohio State"

# 3. Sunday Update: Run automated evaluation, calibration, and next week forecasts
cfb-power sunday-update --season 2026 --completed-week 7 --upcoming-week 8
# Or run offline without calling the CFBD API:
cfb-power sunday-update --season 2026 --completed-week 7 --upcoming-week 8 --offline

# 4. Dataset Management: Activate verified dataset snapshot
cfb-power activate-dataset [DATASET_ID]

# 5. Model Training & Backtesting
cfb-power train

# 6. Workbook Export: Generate formatted XLSX and CSV package
cfb-power export-workbook
```

---

## Web Dashboard Overview

The dashboard includes 7 dedicated tabs:

1. **🏆 Power Rankings**: Sortable table of all FBS teams with overall rating, offensive/defensive splits, tiers, SOS, recent form, and situational stats (YPP, 3rd down rate, turnover margin).
2. **⚔️ Matchup Analyzer**: Interactive selector to match up any two FBS teams with neutral/home selection, projected score, spread, total, win probability, key edge summary, and side-by-side "Tale of the Tape".
3. **📈 Weekly Forecasts**: Published weekly forecast runs with cryptographic SHA-256 manifest verification, slate summary metrics, and export downloads.
4. **🔄 Sunday Weekly Update**: Automated single-click weekly operations: grades previous week's predictions, adaptively recalibrates HFA and margin variance, updates ratings, and prepares next week's slate.
5. **📊 Results Report**: Forecast grading against final scores, straight-up accuracy, ATS margin error, Over/Under MAE, and Brier calibration score.
6. **📥 Ingestion & Revisions**: CFBD batch snapshot ingestion, local raw audit cache inspection, and human-in-the-loop revision diff review.
7. **⚙️ Settings**: CFBD API credential management and cache controls.

---

## Ingestion & Revisions Workflow

To maintain absolute data integrity:

1. **Raw Snapshot Archiving**: Every API query is saved verbatim to `data/raw/cfbd/<season>/<timestamp>/` alongside a request manifest.
2. **Box-Score Caching with Verification**: Historical completed game box-scores are cached locally. If team scores change in revision, the cache is automatically invalidated and re-fetched from the API.
3. **Revision Gatekeeping**: If a subsequent capture of a past week contains score or stat revisions, the ingestion tool flags a `ReviewRequired` state rather than silently overwriting inputs.
4. **Manifest Fingerprinting**: Published datasets in `data/datasets/` contain SHA-256 checksums for each file to ensure tamper resistance.

---

## Project Structure

```text
CFB-Power/
├── src/cfb_power/
│   ├── cli.py                 # Typer CLI application entry point
│   ├── config.py              # Path definitions and environment settings
│   ├── dashboard.py           # Streamlit web application
│   ├── dashboard_power.py     # Power Rankings & Matchup Analyzer UI components
│   ├── dataset_workflow.py    # Snapshot activation and dataset management
│   ├── desktop.py             # Linux desktop launcher daemon
│   ├── exports.py             # Multi-sheet Excel workbook and CSV export writer
│   ├── features.py            # Pregame rolling features and matchup joins
│   ├── model.py               # Ridge regression score models
│   ├── power_ranking.py       # Iterative Ridge rating solver and matchup comparator
│   ├── results_report.py      # Historical forecast evaluation and grading
│   ├── results_ui.py          # Results report dashboard tab
│   ├── revisions.py           # Snapshot diff verification and review logic
│   ├── team_names.py          # FBS team ID and display name resolver
│   ├── upcoming.py            # Next-week prediction pipeline
│   ├── validation.py          # Data contract validation rules
│   ├── weekly_update.py       # Automated Sunday cycle and adaptive calibration
│   └── ingestion/
│       ├── cfbd_batch.py      # Audited CFBD API batch capture and snapshotting
│       ├── cfbd_canonical.py  # Raw JSON to canonical CSV transformer
│       └── cfbd_client.py     # Resilient CFBD HTTP client with rate-limiting
├── tests/                     # 230+ unit, integration, and offline verification tests
├── data/
│   ├── raw/                   # Archived API responses (Git ignored)
│   ├── datasets/              # Immutable verified dataset snapshots (Git ignored)
│   ├── processed/             # Active model inputs (Git ignored)
│   ├── runs/                  # Published weekly forecast runs (Git ignored)
│   └── exports/               # Generated Excel workbooks & CSVs (Git ignored)
├── docs/                      # Specification and design documentation
├── pyproject.toml             # Project metadata, dependencies, and tools
└── README.md
```

---

## Development & Quality Assurance

### Code Standards
The codebase adheres strictly to Ruff linting and formatting standards:

```bash
# Check formatting
ruff format --check src tests

# Auto-format code
ruff format src tests

# Run linter checks
ruff check src tests
```

### Running Tests
The test suite consists of 239+ tests ensuring no data leakage, strict schema contracts, Ridge mathematical consistency, and offline safety:

```bash
pytest
```

---

## License

This project is licensed under the MIT License.
