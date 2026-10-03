# CFBD batch capture and snapshot publication

## Command

From the repository with its environment active and CFBD_API_KEY exported:

```bash
python -m cfb_power.ingestion.cfbd_batch --season 2026 --season-type regular --week 1
```

This is a standalone module CLI, not a new cfb-power subcommand. Season and season type are explicit; supported types are regular and postseason. Week is optional, including week zero. --data-dir defaults to Paths().data_dir. .env is not automatically loaded.

The client makes one schedule request with year, seasonType, classification=fbs, and optional week. It then makes one /games/teams request per selected completed game with id alone. The cost is 1 + N requests, sequentially. The default completed-game cap is 100; larger captures require an explicit --max-completed-games value. The cap is checked after archiving the schedule and before requesting any boxes. There are no automatic retries; rate-limit or network errors fail the run.

## Eligibility

Only FBS-vs-FBS games are published. FBS-FCS games are archived and excluded with non_fbs_matchup. Missing or unsupported classifications fail the run. Upcoming games with TBD or absent kickoff are excluded with unconfirmed_upcoming_kickoff. Unfinished games that already have scores are excluded with unfinished_with_scores. Completed games with invalid or unconfirmed kickoff fail; they are not silently dropped. Empty eligible scopes, out-of-scope responses, and missing or malformed selected boxes fail.

## Storage and safety

Each invocation generates a UTC/UUID run ID. Decoded JSON responses are archived under data/raw/cfbd/{season}/{run_id}/ before mapping, preserving response values rather than original HTTP byte formatting. The raw manifest records request parameters, retrieval timestamps, SHA-256 hashes, scope, counts, exclusions, and the play-count/red-zone policies. It never records request headers or API keys.

Validated CSVs and their hashes are written to a hidden staging directory under data/processed/cfbd/. A same-parent directory rename exposes the completed snapshot at data/processed/cfbd/{run_id}/ only after both tables and a ready manifest exist. Published files are games.csv, team_game_stats.csv, and manifest.json. games.csv adds season_type; team statistics retain offensive_plays_basis. Missing numeric data is serialized as empty CSV fields, not zeros.

Raw status validated means validation completed; the published snapshot's ready manifest is the publication authority. Failure retains captured raw responses and a failed audit containing only the exception type, removes owned staging output, and raises a sanitized BatchIngestionError. Ordinary successful runs never overwrite prior snapshots or active data/processed/games.csv and team_game_stats.csv. This is process-level atomic visibility, not a claim of power-loss durability on every filesystem.

## Acceptance and limitations

Tests use the committed real-response fixtures and mocked API calls. They cover request shape, archive hashes, mapping, exclusions, upcoming behavior, failures, request limits, repeated captures, failed publication cleanup, and the CLI. Run the full repository suite before live validation.

No activation, merging of weekly snapshots into season tables, model training, or forecasting happens here. Snapshot activation and combined historical datasets remain separate work. HTTP headers, raw response bytes, failed-request bodies, API-side completeness guarantees, automatic retries, and a team display-name table are not implemented. Verify one completed week live before expanding scope.
