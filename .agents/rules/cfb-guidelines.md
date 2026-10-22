# CFB-Power Guidelines

Guidelines and constraints for developing, maintaining, and operating the College Football Power Rankings and Forecasting System (`CFB-Power`).

- **Virtual Environment**: Use `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/ruff`.
- **Formatting**: Run `ruff check` and `ruff format` before commits.
- **Security**: Never commit `.env` or print `CFBD_API_KEY` in terminal output or logs.
- **Data Integrity**: Always preserve SHA-256 manifests and raw JSON audits in `data/raw/` and `data/datasets/`.
- **Forecasting Models**: All upcoming weekly game forecasts must default to opponent-adjusted Power Rankings with situational box stats, recency weighting, and calibrated home-field advantage.
- **Testing**: Automated unit tests must remain strictly offline with zero live network calls.
