# Experimental power ratings: milestone 1

## Status

This is an offline source-coverage audit and score-based opponent-adjusted rating engine.
It does not change the dashboard, current rolling-average forecaster, or Results Report.
It makes zero API requests and does not establish account access or provider quota.
Advanced box statistics, recruiting/talent, and player availability are not yet rating inputs.
The audit marks those groups not integrated, not zero.

## Run from the project checkout

Replace DATASET_ID with one assembled dataset directory already on your computer.
Stop active capture/forecast/report jobs before changing branches or updating code.

```bash
.venv/bin/python -m pytest
AS_OF="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
.venv/bin/python -m cfb_power.rating_workflow audit \
  --dataset data/datasets/DATASET_ID --data-dir data \
  --season 2026 --as-of "$AS_OF"
.venv/bin/python -m cfb_power.rating_workflow build \
  --dataset data/datasets/DATASET_ID --data-dir data \
  --season 2026 --as-of "$AS_OF"
```

Each command publishes a new directory under data/rating_snapshots. Audit output includes
coverage.json, coverage.csv, power_ratings.xlsx, and a manifest. Build additionally exports
team_ratings.csv. The workbook contains Coverage and, for build, Team Ratings.
Input datasets and earlier snapshots are not overwritten. No new dependencies are required.
Snapshots use a hidden staging directory and rename on completion; interrupted processes can
leave staging directories. Filesystem crash-durability is not claimed.

## Rating definition

For a listed home team H and away team A:

    expected home points = base + offense[H] - defense[A] + venue / 2
    expected away points = base + offense[A] - defense[H] - venue / 2
    power[team] = offense[team] + defense[team]

Higher defense ratings mean better defense. Offense and defense are centered across the
represented team universe. The base adjusts with centering, preserving fitted scores.
The home-field term is learned jointly and penalized toward zero; neutral games have no
venue term. No positive home advantage is forced. There is no extra schedule-strength bonus.
Schedule strength is the recency-weighted mean of opponents' current power ratings.
Ranks use equal ranks for exact ties and a deterministic team-ID display order.

Weighted penalized least squares estimates offense, defense, base, and home field jointly.
Game weight is 2 ** (-age_in_days / half_life_days). Team penalties pull estimates toward
zero or discounted previous-season estimates. More current data naturally increases its
influence relative to that fixed penalty. Only the immediately preceding season is used
for priors. Its own fit anchors recency at its latest eligible game rather than decaying all
of last season by another year. New/unobserved teams have league-average priors.

Engineering defaults: 35-day half-life, team penalty 4, venue penalty 10, previous-season
carry 0.5, low-history threshold 3 games. These are not tuned or proven optimal.
They are recorded in each build manifest. The Python API accepts RatingSettings overrides.
Do not tune them on final evaluation weeks or interpret training fit as predictive accuracy.

## Coverage and timestamps

Only completed games strictly before the explicit as-of timestamp enter training.
If result_available_at_utc exists, each included result must also have a nonmissing
availability timestamp strictly before the cutoff. This column is optional enrichment;
the current canonical pipeline does not provide it. Without it, kickoff filtering is not
proof the final result was available at the cutoff, especially during a game.
Even with it, the service does not reconstruct historical source revisions or independently
verify those timestamps. The manifest always records point_in_time_provenance_verified=false.
Do not present retrospective runs from revised datasets as verified live pregame records.

The team universe contains current-season teams observed before the cutoff, previous-season
teams with eligible history, and optional explicit team_ids supplied through the Python API.
It is not an independently verified FBS membership registry. CLI outputs use IDs, not names;
full FBS coverage and verified name/roster enrichment are later work. No-history teams are
provisional. Disconnected schedules are flagged; comparisons across components depend on
regularization and priors. The audit denominator is represented, eligible dataset games,
not the full official schedule. Missing results and stale history produce rating warnings.

Rates with zero opportunities remain undefined. Missing red-zone counts stay missing.
Yards per offensive play is explicitly labeled a proxy under the current canonical basis.
The audit does not claim the proxy is a verified snap count or passing yards per attempt.
Basic box metrics are audited for coverage, not silently activated as rating inputs.

## Python matchup API

```python
from cfb_power.power_ratings import build_power_ratings
from cfb_power.rating_workflow import load_rating_dataset

manifest, games, stats, source_hash = load_rating_dataset("data/datasets/DATASET_ID")
snapshot = build_power_ratings(
    games, season=2026, as_of="2026-10-20T03:00:00Z",
)
# Substitute two real team IDs present in snapshot.teams.
# comparison = snapshot.compare(team_a_id, team_b_id, venue="neutral")
```

Venues: neutral, team_a_home, team_b_home. The matchup helper is experimental and separate
from scheduled forecast publication and Sunday scoring. It returns expected points, margin,
total, and warnings, but no calibrated win probability or interval. Negative linear score
estimates raise an error rather than silently clipping and breaking the rating identity.
A command-line compare interface and dashboard integration are later milestones.

## Input integrity and tests

The loader accepts only ready dataset-basic-v1 manifests with exactly the two canonical CSV
artifacts. It verifies hashes and row counts, parses the bytes it hashed, rejects duplicate
CSV headers, and checks team-stat identities/scores against completed games. Current
canonical play-count provenance is required. Artifacts and manifest are checked again
before publication. Local hashes do not protect against deliberate coordinated rewrites.

The two new test modules exercise synthetic fixtures and end-to-end local publication.
During implementation, 57 new tests passed in an isolated sandbox. The pre-existing full
suite, real local datasets, native dashboard, live provider access, and predictive accuracy
were not validated there. Run the full suite locally before merge; if the existing 214-test
baseline is unchanged, the combined suite is expected to contain 271 tests.

## Next acceptance gates

1. Run audit/build on a real local dataset and review missing results and coverage.
2. Verify the workbook, input preservation, and recorded model/settings/hashes.
3. Verify provider entitlements, historical feature coverage, and enrichment timestamps.
4. Add strictly chronological baseline comparisons and calibrate probability/uncertainty.
5. Enable richer statistics and UI integration only after separate tests and review.
