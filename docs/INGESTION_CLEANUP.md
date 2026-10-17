# HTTP diagnostics and development probe

HTTP failures now raise CfbdHttpError with numeric status_code and an allowlisted endpoint. Response bodies, reasons, headers, query strings, and unknown endpoint text are not included in this exception's message. Batch ingestion preserves those controlled fields in a failed raw audit and reports the audit path. Other exceptions remain type-only; failed runs do not publish snapshots. There are no automatic retries.

Run the optional development probe from the installed project:

```bash
python -m cfb_power.probe --season 2026 --season-type regular --game-id 401871090
```

The scripts/probe_cfbd.py entry point accepts the same arguments. Season is required; game ID is optional. Without an ID the probe chooses the lowest-ID eligible completed FBS-vs-FBS game in scope. It makes at most three API requests (schedule, ID-only team box score, advanced box score). Check quota before running.

The probe reads the project-root .env explicitly, with saved-key precedence over the environment. It does not change CLI ingestion key loading. Each run writes to a unique project-root data/probes/cfbd directory, never model inputs. A failure may leave partial diagnostic outputs; they are not published snapshots. It validates the returned game ID, team identities, and nested team stat lists before archiving the traditional box score.

Generated probes, datasets, forecast runs, and egg-info are ignored by Git. This does not delete existing files or untrack already committed files. Scripts remain trackable. No real credentials or captured payloads are included.

Validation: 18 targeted tests passed offline in an isolated sandbox; two .env tests were not executed because python-dotenv is unavailable there. Config/canonical normalization imports were stubbed; batch tests exercise pre-normalization failure paths. Run the full repository suite locally before merge. With the previous 134 tests and 20 added cases, expect 154 tests if otherwise unchanged. No live API requests were made during testing.
