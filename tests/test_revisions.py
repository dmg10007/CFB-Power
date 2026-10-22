"""Hash-verified revision fixtures; no network calls."""

import json
from unittest.mock import Mock

import pandas as pd
import pytest

from cfb_power import dataset_workflow as dw
from cfb_power.revisions import ReviewRequired, assemble_or_review, inspect_snapshots, resume_review
from cfb_power.schemas import GAME_COLUMNS, TEAM_GAME_STAT_COLUMNS


def snapshot(tmp_path, name, conversions=3, attempts=9, score=34, bad=False):
    path = tmp_path / name
    path.mkdir()
    games, stats = [], []
    for gid in (401871090, 999):
        game = dict(
            zip(
                GAME_COLUMNS,
                [
                    gid,
                    2026,
                    6,
                    "2026-10-03T16:00:00Z",
                    2653,
                    2572,
                    55,
                    score if gid == 401871090 else 34,
                    False,
                    True,
                ],
                strict=True,
            )
        )
        games.append(game)
        for team, opp, home in ((2653, 2572, True), (2572, 2653, False)):
            row = {column: 1 for column in TEAM_GAME_STAT_COLUMNS}
            row.update(
                game_id=gid,
                team_id=team,
                opponent_id=opp,
                is_home=home,
                points=game["home_score"] if home else game["away_score"],
                points_allowed=game["away_score"] if home else game["home_score"],
                offensive_plays=66,
                third_down_attempts=attempts if gid == 401871090 else 9,
                third_down_conversions=conversions if gid == 401871090 else 3,
                offensive_plays_basis=dw.PLAY_COUNT_BASIS,
            )
            if bad and gid == 401871090 and team == 2572:
                row["points"] = 90
            stats.append(row)
    outputs = []
    for filename, rows in (("games.csv", games), ("team_game_stats.csv", stats)):
        pd.DataFrame(rows).to_csv(path / filename, index=False)
        outputs.append({"file": filename, "rows": len(rows), "sha256": dw._sha256(path / filename)})
    dw._write_json(
        path / "manifest.json",
        {
            "status": "ready",
            "schema_version": dw.SNAPSHOT_SCHEMA,
            "run_id": name,
            "started_at_utc": f"2026-10-{9 if name == 'old' else 17:02d}T01:00:00Z",
            "offensive_plays_basis": dw.PLAY_COUNT_BASIS,
            "outputs": outputs,
        },
    )
    return path


@pytest.fixture
def pair(tmp_path):
    return [snapshot(tmp_path, "old"), snapshot(tmp_path, "new", 2, 8)]


def build(pair, tmp_path, decisions=None, token=None):
    return dw.assemble_dataset(
        pair,
        tmp_path,
        revision_decisions=decisions or {"401871090": "new"},
        review_token=token or inspect_snapshots(pair)["token"],
    )


def test_review_collects_versions(pair):
    plan = inspect_snapshots(pair)
    assert set(plan["conflicts"]) == {"401871090"}
    assert [v["run_id"] for v in plan["conflicts"]["401871090"]] == ["old", "new"]


def test_strict_assembly_rejects(pair, tmp_path):
    with pytest.raises(dw.DatasetError, match="conflicting team statistics"):
        dw.assemble_dataset(pair, tmp_path)
    assert not (tmp_path / "datasets").exists()


def test_review_pause(pair, tmp_path):
    with pytest.raises(ReviewRequired):
        assemble_or_review(pair, tmp_path)
    assert not (tmp_path / "datasets").exists()


def test_selection_provenance_and_immutability(pair, tmp_path):
    before = {p: p.read_bytes() for path in pair for p in path.iterdir()}
    result = build(pair, tmp_path)
    manifest, games, stats = dw._load_verified(result, dw.DATASET_SCHEMA, "dataset")
    row = stats[(stats.game_id == 401871090) & (stats.team_id == 2572)].iloc[0]
    assert row.third_down_attempts == 8 and row.third_down_conversions == 2
    assert len(games) == 2 and len(stats) == 4
    assert set(stats[stats.game_id == 999].third_down_attempts) == {9}
    assert manifest["revisions"]["decisions"] == {"401871090": "new"}
    assert len(manifest["revisions"]["reviewed_conflicts"]["401871090"]) == 2
    assert all(p.read_bytes() == value for p, value in before.items())


@pytest.mark.parametrize(
    "decisions", [{}, {"401871090": "missing"}, {"401871090": "new", "999": "old"}]
)
def test_invalid_decisions(pair, tmp_path, decisions):
    with pytest.raises(ValueError):
        dw.assemble_dataset(
            pair,
            tmp_path,
            revision_decisions=decisions,
            review_token=inspect_snapshots(pair)["token"],
        )
    assert not (tmp_path / "datasets").exists()


def test_new_input_invalidates_approval(pair, tmp_path):
    token = inspect_snapshots(pair)["token"]
    pair.append(snapshot(tmp_path, "extra", 2, 8))
    with pytest.raises(ValueError, match="changed"):
        build(pair, tmp_path, token=token)


def test_corrupt_artifact(pair):
    with (pair[1] / "team_game_stats.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(dw.DatasetError, match="hash mismatch"):
        inspect_snapshots(pair)


def test_inconsistent_selected_score(tmp_path):
    paths = [snapshot(tmp_path, "old"), snapshot(tmp_path, "new", 2, 8, bad=True)]
    with pytest.raises(dw.DatasetError, match="disagree"):
        build(paths, tmp_path)


def test_complete_game_score_revision(tmp_path):
    paths = [snapshot(tmp_path, "old"), snapshot(tmp_path, "new", 2, 8, score=35)]
    _, games, stats = dw._load_verified(build(paths, tmp_path), dw.DATASET_SCHEMA, "dataset")
    assert games.set_index("game_id").loc[401871090, "away_score"] == 35
    assert (
        stats[(stats.game_id == 401871090) & (stats.team_id == 2653)].iloc[0].points_allowed == 35
    )


def test_identity_change_rejected(pair, tmp_path):
    path = pair[1]
    frame = pd.read_csv(path / "games.csv")
    frame.loc[frame.game_id == 401871090, "away_team_id"] = 123
    frame.to_csv(path / "games.csv", index=False)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["outputs"][0]["sha256"] = dw._sha256(path / "games.csv")
    dw._write_json(path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="identities"):
        build(pair, tmp_path)


def test_resume_uses_existing_inputs(pair, tmp_path, monkeypatch):
    from cfb_power import upcoming

    monkeypatch.setattr(dw, "discover_snapshots", lambda data_dir: pair)
    forecast = Mock(return_value="forecast-run")
    monkeypatch.setattr(upcoming, "predict_upcoming", forecast)
    assert (
        resume_review(tmp_path, {"401871090": "new"}, inspect_snapshots(pair)["token"])
        == "forecast-run"
    )
    forecast.assert_called_once()
