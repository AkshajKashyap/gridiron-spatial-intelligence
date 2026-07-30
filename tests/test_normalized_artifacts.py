import hashlib
import json

import pandas as pd
import pytest

import gridiron_spatial.normalized_artifacts as artifacts
from gridiron_spatial.coordinate_frame import add_normalized_coordinates
from gridiron_spatial.normalized_artifacts import (
    NormalizedArtifactError,
    write_normalized_tracking_release,
)
from gridiron_spatial.normalized_tracking import (
    NORMALIZED_ENTITY_FRAME_SCHEMA,
    freeze_normalized_entity_frames,
    reconcile_normalized_entity_frames,
)


def _built(week, *, direction="right"):
    week_number = int(week[-2:])
    raw = pd.DataFrame(
        {
            "game_id": [f"game-{week}", f"game-{week}"],
            "play_id": ["1", "1"],
            "phase": ["input", "output"],
            "frame_id": [1, 1],
            "nfl_id": ["101", "101"],
            "week": [week, week],
            "week_number": [week_number, week_number],
            "split": ["development_train", "development_train"],
            "player_side": ["Offense", "Offense"],
            "player_role": ["Targeted Receiver", "Targeted Receiver"],
            "player_position": ["WR", "WR"],
            "player_to_predict": [True, True],
            "play_direction": [direction, direction],
            "x": [10.0, 11.0],
            "y": [20.0, 20.0],
            "dir": [0.0, float("nan")],
            "o": [0.0, float("nan")],
        }
    )
    normalized = freeze_normalized_entity_frames(
        add_normalized_coordinates(raw)
    )
    return {
        "normalized_frame": normalized,
        "reconciliation": reconcile_normalized_entity_frames(raw, normalized),
        "input_rows": 1,
        "output_rows": 1,
        "runtime_seconds": 0.25,
    }


def test_partitioned_release_order_manifest_checksums_totals_and_no_mutation(
    tmp_path,
):
    weeks = ("2023_w02", "2023_w03")
    built = {
        "2023_w02": _built("2023_w02"),
        "2023_w03": _built("2023_w03", direction="left"),
    }
    snapshots = {
        week: result["normalized_frame"].copy(deep=True)
        for week, result in built.items()
    }
    execution_order = []

    def builder(week):
        execution_order.append(week)
        return built[week]

    destination = tmp_path / "normalized"
    manifest = write_normalized_tracking_release(
        destination, weeks, builder
    )

    assert execution_order == list(weeks)
    assert {path.name for path in destination.iterdir()} == {
        "normalized_2023_w02.parquet",
        "normalized_2023_w03.parquet",
        "manifest.json",
    }
    assert manifest["requested_weeks"] == list(weeks)
    assert manifest["processed_weeks"] == list(weeks)
    assert [item["week"] for item in manifest["partitions"]] == list(weeks)
    assert manifest["aggregate"]["input_rows"] == 2
    assert manifest["aggregate"]["output_rows"] == 2
    assert manifest["aggregate"]["combined_rows"] == 4
    assert manifest["aggregate"]["unique_games"] == 2
    assert manifest["aggregate"]["unique_plays"] == 2
    for item in manifest["partitions"]:
        path = destination / item["relative_filename"]
        assert item["combined_rows"] == 2
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        assert list(pd.read_parquet(path).columns) == (
            NORMALIZED_ENTITY_FRAME_SCHEMA
        )
    assert json.loads((destination / "manifest.json").read_text()) == manifest
    assert str(tmp_path) not in json.dumps(manifest)
    for week in weeks:
        pd.testing.assert_frame_equal(
            built[week]["normalized_frame"], snapshots[week]
        )


def test_overwrite_protection_and_later_failure_preserve_destination(tmp_path):
    destination = tmp_path / "normalized"
    write_normalized_tracking_release(
        destination, ("2023_w02",), lambda week: _built(week)
    )
    original_manifest = (destination / "manifest.json").read_bytes()
    called = False

    def forbidden_builder(week):
        nonlocal called
        called = True
        return _built(week)

    with pytest.raises(FileExistsError):
        write_normalized_tracking_release(
            destination, ("2023_w03",), forbidden_builder
        )
    assert not called

    def fail_later(week):
        if week == "2023_w03":
            raise RuntimeError("synthetic later-week failure")
        return _built(week)

    with pytest.raises(RuntimeError, match="later-week"):
        write_normalized_tracking_release(
            destination,
            ("2023_w02", "2023_w03"),
            fail_later,
            overwrite=True,
        )
    assert (destination / "manifest.json").read_bytes() == original_manifest
    assert not list(tmp_path.glob(".normalized.tmp-*"))

    write_normalized_tracking_release(
        destination,
        ("2023_w03",),
        lambda week: _built(week),
        overwrite=True,
    )
    assert not (destination / "normalized_2023_w02.parquet").exists()
    assert (destination / "normalized_2023_w03.parquet").exists()


@pytest.mark.parametrize("failure", ["reconciliation", "duplicate"])
def test_validation_failure_prevents_release(tmp_path, failure):
    built = _built("2023_w02")
    if failure == "reconciliation":
        built["reconciliation"] = dict(built["reconciliation"])
        built["reconciliation"]["status"] = "FAIL"
    else:
        built["normalized_frame"] = pd.concat(
            [built["normalized_frame"], built["normalized_frame"].iloc[[0]]],
            ignore_index=True,
        )
    destination = tmp_path / failure
    with pytest.raises(NormalizedArtifactError):
        write_normalized_tracking_release(
            destination, ("2023_w02",), lambda week: built
        )
    assert not destination.exists()


def test_readback_schema_mismatch_prevents_release(tmp_path, monkeypatch):
    built = _built("2023_w02")
    real_read = pd.read_parquet

    def broken_readback(path):
        return real_read(path).drop(columns="o_norm")

    monkeypatch.setattr(artifacts.pd, "read_parquet", broken_readback)
    destination = tmp_path / "schema-failure"
    with pytest.raises(NormalizedArtifactError, match="read-back schema"):
        write_normalized_tracking_release(
            destination, ("2023_w02",), lambda week: built
        )
    assert not destination.exists()
