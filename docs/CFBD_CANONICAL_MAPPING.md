# Canonical CFBD mapping: first slice

## Usage

`cfb_power.ingestion.cfbd_canonical.normalize_cfbd_canonical(games_payload, team_stats_payload)` returns two pandas DataFrames. It is pure: no API calls, file writes, or credential access.

Games have GAME_COLUMNS. Completed-game stats have TEAM_GAME_STAT_COLUMNS plus offensive_plays_basis. IDs are positive source-owned integers, never display names. This slice requires FBS-vs-FBS games and rejects mixed-classification records rather than silently filtering them. Kickoff must be timezone-aware and confirmed. Weeks may be zero. Upcoming games require missing scores and have no team-stat rows; this does not yet implement upcoming prediction features.

## Approved missing-data policy

Offensive plays are a proxy: passing attempts (the second value of completionAttempts) plus rushingAttempts. offensive_plays_basis records passing_attempts_plus_rushing_attempts_proxy. It is not a verified snap count and is not interchangeable with advanced PPA plays. Both inputs must be present and the sum positive.

Red-zone attempts and touchdowns remain missing, never zero. Red-zone rate is removed from ROLLING_METRICS; the canonical columns remain for later enrichment. Rebuild features and retrain models after this feature-list change. Scoring opportunities are not substituted for red-zone attempts.

## Mappings and validation

Schedule id, season, week, startDate, homeId, awayId, homePoints, awayPoints, neutralSite, and completed map to the existing game schema. Postgame probabilities and Elo values are not used as features.

Basic totalYards, netPassingYards, rushingYards, and turnovers are parsed as integers. Negative net yardage is allowed; counts are nonnegative. thirdDownEff and completionAttempts use successes-attempts pairs. totalPenaltiesYards uses penalties-yards. possessionTime is converted from minutes:seconds, with seconds below 60.

Opponent ID and points allowed come from the schedule. Takeaways equal the opponent's reported turnovers, not fumblesRecovered. Every completed game must have exactly one box with exactly two matching team IDs, correct homeAway roles, and matching scores. Duplicate games, boxes, team IDs, and stat categories fail. Missing required stats and inconsistent total/passing/rushing yards also fail. Unknown stat categories are retained only in raw fixtures, not canonical outputs.

## Fixtures and scope

canonical_games_401856766.json and canonical_team_stats_401856766.json preserve the user-supplied response values for North Carolina at TCU, game 401856766, received 2026-10-02. Only JSON formatting is changed. The game is marked neutral-site. The play-count proxies are 57 and 71; possession seconds are 1618 and 1982 respectively. Neither fixture contains red-zone categories.

This commit provides mapping, fixtures, tests, and the approved rolling-feature adjustment only. Batch API retrieval, raw archival, ingestion audit logs, CLI wiring, durable output publication, team display-name tables, season-type tracking, and upcoming feature construction are separate follow-ups. The downstream batch layer must archive source responses and carry offensive_plays_basis into audit metadata before publishing a dataset.
