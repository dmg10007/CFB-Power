# CFB-Power Project Guidelines

Guidelines and constraints for developing, maintaining, and operating the College Football Power Rankings and Forecasting System (`CFB-Power`).

---

## 1. Environment & Tooling

- **Virtual Environment**: Always use the project's virtual environment at `.venv` (`.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/ruff`).
- **Linter & Formatter**: Code must comply with Ruff rules:
  - Run `.venv/bin/ruff check src tests` to lint.
  - Run `.venv/bin/ruff format src tests` to format.
- **Python Version**: Python 3.12+ features (modern type unions `X | Y`, built-in generics `list[str]`, etc.).

---

## 2. Security & Credentials

- **API Keys**: Never hardcode API keys, secrets, or bearer tokens in code, tests, or documentation.
- **Environment Variables**: Load secrets from `.env` using `python-dotenv`. Specifically:
  - `CFBD_API_KEY`: CollegeFootballData API bearer key.
- **Logging & Errors**: Never print full URLs containing query parameters or headers with auth tokens in terminal outputs, audit manifests, or exception strings.

---

## 3. Data Integrity & Ingestion

- **Auditability**: All raw API responses must be captured into immutable JSON archives in `data/raw/` with a corresponding `manifest.json`.
- **SHA-256 Hashes**: Every published snapshot and dataset must calculate and record SHA-256 hashes for all output CSV files (`games.csv`, `team_game_stats.csv`).
- **Revision Protection**: If multiple snapshots contain differing statistics for the same `game_id`, trigger revision inspection (`assemble_or_review`) rather than silently overwriting.
- **Offline Reliability**: Batch ingestion and weekly update workflows must provide graceful fallbacks or `--offline` flags when CFBD API rate limits or quota caps are hit.

---

## 4. Modeling & Forecasting Standards

- **Opponent Adjustments**: All weekly forecasts must use opponent-adjusted Power Rankings (Ridge regression ratings, SOS adjustments, situational box statistics, exponential recency weighting, and calibrated home-field advantage).
- **Backward Compatibility**: Preserve support for the walk-forward benchmark model (`model="rolling_average"`).
- **Outputs Contract**: Every predictions run must output:
  - `home_expected_points` & `away_expected_points`
  - `projected_spread_home` & `projected_total`
  - `home_win_probability`
  - Uncertainty score bounds (`home_score_low`, `home_score_high`, `away_score_low`, `away_score_high`)
  - `confidence_tier` (`High`, `Medium`, `Low`, or `Provisional`)
  - Complete `manifest.json` recording model settings and dataset provenance.

---

## 5. Testing Protocols

- **Zero Network Calls in Automated Tests**: All unit and integration tests must run fully offline using synthetic fixtures or mock responses. Never make live HTTP requests inside `pytest`.
- **Test Execution**: Verify changes across the test suite before committing:
  ```bash
  .venv/bin/pytest
  ```
- **CLI Tests**: When testing CLI commands with Typer, use `monkeypatch.delenv("CFBD_API_KEY", raising=False)` to prevent picking up active environment keys.
