# CFBD fixture provenance and regression coverage

## Real-response fixture

`tests/fixtures/cfbd/advanced_box_real_401856766_trimmed.json` is derived from the user's attached `raw.json` contents, received on 2026-10-02. The response identifies game 401856766, season 2026, week 1, North Carolina at TCU. The identifiers come from the passing and rushingAdvanced entries; gameInfo has no game ID.

This is a trimmed real-response fixture, not the full response. All retained values are copied without alteration. It retains gameInfo and all ten team categories. The players section is omitted because the normalizer handles only teams.

For passing offense and defense, it retains ppa, attempts, totalYards, completions, successRate, unknown-location totalYards, and deep-right ppa and attempts. For rushingAdvanced offense and defense, it retains ppa, attempts, sacks, totalRushingYards, and left-direction ppa, carries, and yards. Other categories and team-level metadata are retained in full.

The fixture therefore contains 110 scalar fields per team, not the full response's 620. The user's earlier live validation reported preservation of all 1,240 scalar fields across both teams. Offline tests of this fixture cover only the 220 retained scalar fields and do not claim full-response coverage or independent verification of the API's statistics.

## Synthetic fixture

`tests/fixtures/cfbd/advanced_box_verified_shape.json` is synthetic Alpha/Beta test data, despite its legacy filename. It is not a verified API response. Its filename and existing tests are retained unchanged in this commit. Use the separate real-response fixture for captured-response regression coverage. This document supplements the original schema notes, which describe the initial assumed fixture.

## Tests

`tests/test_cfbd_client.py` contains two offline tests: delegation with the id query parameter, and URL, authorization header, accept header, timeout, and response decoding using mocked network access. The test key is a dummy value; no real key or network access is required.

`tests/test_cfbd_real_fixture.py` contains five tests: team and identifier checks with trimmed field counts; selected known values; null, zero, and legacy spaced keys; order independence; and preservation of every retained scalar with collision detection.

The seven new tests passed in an isolated sandbox using the client and normalizer code from the conversation. Run `python -m pytest` in the actual repository before opening the pull request.

## Known limitations

The current normalizer converts camelCase but does not replace spaces. Thus the real location key `deep right` produces a column such as `passing_offense_locations_deep right_ppa`. The regression test records existing behavior; this commit does not change column naming. A future naming change must update consumers and these tests together.

KNOWN_CATEGORIES does not yet list passing or rushingAdvanced, but the normalizer dynamically processes both. Lists within metric objects are ignored. The provided game_id is copied into output rows without validation against response metadata. These are follow-up hardening items, not fixed by this test-only commit.
