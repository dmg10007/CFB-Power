# Verified datasets and explicit-dataset training

## Workflow

1. Capture weekly snapshots (each is immutable and verified by hash):

```bash
for week in 1 2 3 4 5; do
  python -m cfb_power.ingestion.cfbd_batch --season 2026 --season-type regular --week "$week" || break
done
```

Choose only weeks CFBD has populated. Re-capturing a week is safe: identical records are deduplicated.

2. Assemble snapshots into a new dataset:

```bash
python -m cfb_power.dataset_workflow assemble --all
# or choose snapshots explicitly:
python -m cfb_power.dataset_workflow assemble --snapshot data/processed/cfbd/RUN_A --snapshot data/processed/cfbd/RUN_B
```

3. Train from one explicit dataset:

```bash
python -m cfb_power.dataset_workflow train --dataset data/datasets/DATASET_ID
```

--data-dir defaults to the project data directory. .env is not loaded automatically.

## Safety rules

- A snapshot must have status ready, schema canonical-basic-v1, exactly games.csv and team_game_stats.csv, matching SHA-256 hashes, and matching row counts. Tampered, unready, unreadable, or wrong-schema inputs fail the whole assembly.
- --all uses every published snapshot under data/processed/cfbd. Use --snapshot when stale or experimental captures should be excluded. Duplicate paths and run IDs are rejected.
- Snapshots are processed chronologically by manifest start time. Identical records are deduplicated. A completed game may replace an earlier upcoming record for the same game_id (counted as upcoming_replaced_by_completed). Any other difference, including changed scores or statistics, is a conflict and fails; nothing is published.
- Completed games must have exactly two team records matching the schedule's team IDs, home/away roles, opponents, points, and points allowed. Team records for non-completed games are rejected. The play-count proxy basis must be consistent.
- A dataset is published under data/datasets/{dataset_id} through a same-parent staging rename. Its manifest records input run IDs, paths, manifest hashes, scopes, counts, output hashes, the play-count basis, and the red-zone policy. Existing snapshots, previous datasets, and active data/processed CSVs are not modified. The rename gives process-level visibility, not a power-loss durability claim.

## Training behavior

Training verifies the dataset again, validates canonical tables, then reuses the existing feature builder, Ridge models, evaluator, and Excel exporter. It writes only to data/runs/{dataset_id}/{run_id}/: exports/, models/, and run_manifest.json. It never overwrites the active data/processed CSVs or the default export/model directories. Selecting an earlier dataset ID is the rollback mechanism.

Training requires completed games from at least two season/week groups; the existing model requires at least 20 completed matchups. A one-week dataset cannot train because rolling features need earlier games.

Run metrics are in-sample pipeline checks. The model trains on completed games and then predicts them, so run_manifest.json labels evaluation_scope as in_sample_pipeline_check. Do not describe these metrics as forecasting accuracy.

## Limitations

This workflow does not yet build upcoming-matchup features, perform chronological walk-forward evaluation, add team display names, or create dashboard views. Upcoming games are retained in the dataset but are not a complete forecast workflow. CFBD completeness, API limits, and retry behavior are not guaranteed by this layer. Snapshot discovery is by directory convention, and --all can include captures you no longer intend to use.
