import copy
import time

import numpy as np
import pandas as pd
import pytest

from gridiron_spatial.coordinate_frame import add_normalized_coordinates
from gridiron_spatial.normalized_tracking import (
    NORMALIZED_ENTITY_FRAME_KEY,
    NORMALIZED_ENTITY_FRAME_SCHEMA,
    freeze_normalized_entity_frames,
    reconcile_normalized_entity_frames,
)


def _fixture():
    raw = pd.DataFrame(
        {
            "game_id": ["1", "1", "2", "2"],
            "play_id": ["10", "10", "20", "20"],
            "phase": ["input", "output", "input", "output"],
            "frame_id": [1, 1, 1, 1],
            "nfl_id": ["101", "101", "202", "202"],
            "week": ["2023_w01", "2023_w01", "2023_w13", "2023_w13"],
            "week_number": [1, 1, 13, 13],
            "split": [
                "development_train",
                "development_train",
                "validation",
                "validation",
            ],
            "player_side": ["Offense", "Offense", "Defense", "Defense"],
            "player_role": ["Targeted Receiver"] * 2
            + ["Defensive Coverage"] * 2,
            "player_position": ["WR", "WR", "CB", "CB"],
            "player_to_predict": [True, True, True, True],
            "play_direction": ["right", "right", "left", "left"],
            "x": [10.0, 120.26, 122.0, 70.0],
            "y": [20.0, 52.12, 10.0, 30.0],
            "dir": [360.0, -10.0, np.nan, 725.0],
            "o": [-10.0, 725.0, 90.0, np.nan],
        },
        index=[5, 2, 9, 4],
    )
    normalized = add_normalized_coordinates(raw)
    return raw, normalized


def test_freeze_enforces_schema_dtypes_phase_keys_and_no_mutation():
    raw, normalized = _fixture()
    normalized["ignored_extra"] = "drop-me"
    before = normalized.copy(deep=True)

    frozen = freeze_normalized_entity_frames(normalized)

    assert list(frozen.columns) == NORMALIZED_ENTITY_FRAME_SCHEMA
    assert frozen.index.tolist() == [5, 2, 9, 4]
    assert str(frozen["game_id"].dtype) == "string"
    assert str(frozen["phase"].dtype) == "string"
    assert str(frozen["frame_id"].dtype) == "Int64"
    assert str(frozen["week_number"].dtype) == "Int64"
    assert str(frozen["player_to_predict"].dtype) == "boolean"
    assert str(frozen["x"].dtype) == "Float64"
    assert str(frozen["dir_norm"].dtype) == "Float64"
    assert not frozen.duplicated(NORMALIZED_ENTITY_FRAME_KEY).any()
    assert frozen.loc[5, "frame_id"] == frozen.loc[2, "frame_id"] == 1
    assert frozen.loc[5, "phase"] == "input"
    assert frozen.loc[2, "phase"] == "output"
    pd.testing.assert_frame_equal(normalized, before)
    assert raw.index.equals(frozen.index)


def test_freeze_rejects_duplicates_missing_columns_and_unsupported_values():
    _, normalized = _fixture()
    duplicate = pd.concat(
        [normalized, normalized.iloc[[0]]], ignore_index=True
    )
    with pytest.raises(ValueError, match="Duplicate phase-qualified"):
        freeze_normalized_entity_frames(duplicate)
    with pytest.raises(ValueError, match="Missing frozen"):
        freeze_normalized_entity_frames(
            normalized.drop(columns="normalized_coordinate_class")
        )

    cases = [
        ("phase", "future", "phase"),
        ("raw_coordinate_class", "repaired", "raw_coordinate_class"),
        ("coordinate_transform_version", "v2", "transform_version"),
        ("coordinate_transform_applied", "mirrored", "transform_applied"),
    ]
    for column, value, message in cases:
        invalid = normalized.copy(deep=True)
        invalid.loc[invalid.index[0], column] = value
        with pytest.raises(ValueError, match=message):
            freeze_normalized_entity_frames(invalid)


def test_successful_reconciliation_is_complete_deterministic_and_nonmutating():
    raw, normalized = _fixture()
    frozen = freeze_normalized_entity_frames(normalized)
    raw_before = raw.copy(deep=True)
    normalized_before = frozen.copy(deep=True)

    result = reconcile_normalized_entity_frames(raw, frozen)

    assert list(result) == [
        "status",
        "raw_rows",
        "normalized_rows",
        "raw_unique_keys",
        "normalized_unique_keys",
        "missing_normalized_keys",
        "unexpected_normalized_keys",
        "duplicate_raw_keys",
        "duplicate_normalized_keys",
        "raw_field_mismatch_counts",
        "index_preserved",
        "row_order_preserved",
        "coordinate_class_counts",
        "transform_applied_counts",
        "unsupported_phase_values",
        "unsupported_coordinate_class_values",
        "unsupported_transform_versions",
        "unsupported_transform_applied_values",
        "transform_direction_mismatches",
        "invalid_raw_to_nominal_normalized",
    ]
    assert result["status"] == "PASS"
    assert result["raw_rows"] == result["normalized_rows"] == 4
    assert result["raw_unique_keys"] == result["normalized_unique_keys"] == 4
    assert result["missing_normalized_keys"] == []
    assert result["unexpected_normalized_keys"] == []
    assert result["duplicate_raw_keys"] == 0
    assert result["duplicate_normalized_keys"] == 0
    assert result["raw_field_mismatch_counts"] == {
        "x": 0,
        "y": 0,
        "dir": 0,
        "o": 0,
        "play_direction": 0,
        "phase": 0,
    }
    assert result["index_preserved"] is True
    assert result["row_order_preserved"] is True
    assert result["coordinate_class_counts"] == {
        "raw_coordinate_class": {
            "nominal": 2,
            "extended_tolerance": 1,
            "invalid": 1,
        },
        "normalized_coordinate_class": {
            "nominal": 2,
            "extended_tolerance": 1,
            "invalid": 1,
        },
    }
    assert result["transform_applied_counts"] == {
        "identity_right": 2,
        "rotate_180_left": 2,
    }
    assert result == reconcile_normalized_entity_frames(raw, frozen)
    pd.testing.assert_frame_equal(raw, raw_before)
    pd.testing.assert_frame_equal(frozen, normalized_before)


def test_reconciliation_reports_missing_unexpected_raw_and_order_failures():
    raw, normalized = _fixture()
    frozen = freeze_normalized_entity_frames(normalized)

    missing = reconcile_normalized_entity_frames(raw, frozen.iloc[1:])
    assert missing["status"] == "FAIL"
    assert missing["missing_normalized_keys"] == [
        {
            "game_id": "1",
            "play_id": "10",
            "phase": "input",
            "frame_id": 1,
            "nfl_id": "101",
        }
    ]

    invented_row = frozen.iloc[[0]].copy()
    invented_row["game_id"] = "9"
    unexpected_frame = pd.concat([frozen, invented_row], ignore_index=True)
    unexpected = reconcile_normalized_entity_frames(raw, unexpected_frame)
    assert unexpected["status"] == "FAIL"
    assert unexpected["unexpected_normalized_keys"][0]["game_id"] == "9"

    changed = frozen.copy(deep=True)
    changed.loc[5, "x"] = 11.0
    mismatch = reconcile_normalized_entity_frames(raw, changed)
    assert mismatch["status"] == "FAIL"
    assert mismatch["raw_field_mismatch_counts"]["x"] == 1

    shuffled = frozen.iloc[::-1]
    order = reconcile_normalized_entity_frames(raw, shuffled)
    assert order["status"] == "FAIL"
    assert order["row_order_preserved"] is False


def test_reconciliation_detects_duplicate_and_incorrect_provenance():
    raw, normalized = _fixture()
    frozen = freeze_normalized_entity_frames(normalized)

    duplicate = pd.concat([frozen, frozen.iloc[[0]]], ignore_index=True)
    duplicate_result = reconcile_normalized_entity_frames(raw, duplicate)
    assert duplicate_result["status"] == "FAIL"
    assert duplicate_result["duplicate_normalized_keys"] == 2

    incorrect = frozen.copy(deep=True)
    incorrect.loc[9, "coordinate_transform_applied"] = "identity_right"
    result = reconcile_normalized_entity_frames(raw, incorrect)
    assert result["status"] == "FAIL"
    assert result["transform_direction_mismatches"] == 1

    invalid_promoted = frozen.copy(deep=True)
    invalid_promoted.loc[9, "normalized_coordinate_class"] = "nominal"
    result = reconcile_normalized_entity_frames(raw, invalid_promoted)
    assert result["status"] == "FAIL"
    assert result["invalid_raw_to_nominal_normalized"] == 1


def test_vectorized_raw_field_reconciliation_scales_and_is_null_safe():
    row_count = 100_000
    positions = np.arange(row_count)
    direction_angles = np.mod(positions.astype(float), 360.0)
    direction_angles[::10_000] = np.nan
    raw = pd.DataFrame(
        {
            "game_id": np.full(row_count, "1", dtype=object),
            "play_id": np.full(row_count, "10", dtype=object),
            "phase": np.where(positions % 2 == 0, "input", "output"),
            "frame_id": positions // 2 + 1,
            "nfl_id": np.full(row_count, "101", dtype=object),
            "week": np.full(row_count, "2023_w01", dtype=object),
            "week_number": np.ones(row_count, dtype=int),
            "split": np.full(
                row_count, "development_train", dtype=object
            ),
            "player_side": np.full(row_count, "Offense", dtype=object),
            "player_role": np.full(
                row_count, "Targeted Receiver", dtype=object
            ),
            "player_position": np.full(row_count, "WR", dtype=object),
            "player_to_predict": np.ones(row_count, dtype=bool),
            "play_direction": np.where(
                positions % 2 == 0, "right", "left"
            ),
            "x": 10.0 + np.mod(positions, 100) / 10.0,
            "y": np.full(row_count, 20.0),
            "dir": direction_angles,
            "o": np.full(row_count, np.nan),
        }
    )
    normalized = freeze_normalized_entity_frames(
        add_normalized_coordinates(raw)
    )
    raw_before = raw.copy(deep=True)
    normalized_before = normalized.copy(deep=True)

    started = time.perf_counter()
    matching = reconcile_normalized_entity_frames(raw, normalized)
    elapsed = time.perf_counter() - started

    assert elapsed < 10.0
    assert matching["status"] == "PASS"
    assert matching["raw_field_mismatch_counts"] == {
        "x": 0,
        "y": 0,
        "dir": 0,
        "o": 0,
        "play_direction": 0,
        "phase": 0,
    }

    changed_x = normalized.copy(deep=True)
    changed_x.loc[1, "x"] += 1.0
    assert reconcile_normalized_entity_frames(
        raw, changed_x
    )["raw_field_mismatch_counts"]["x"] == 1

    changed_direction = normalized.copy(deep=True)
    changed_direction.loc[1, "play_direction"] = "right"
    assert reconcile_normalized_entity_frames(
        raw, changed_direction
    )["raw_field_mismatch_counts"]["play_direction"] == 1

    one_sided_null = normalized.copy(deep=True)
    one_sided_null.loc[0, "dir"] = 1.0
    assert reconcile_normalized_entity_frames(
        raw, one_sided_null
    )["raw_field_mismatch_counts"]["dir"] == 1

    shuffled = normalized.sample(frac=1.0, random_state=7)
    shuffled_result = reconcile_normalized_entity_frames(raw, shuffled)
    assert not any(shuffled_result["raw_field_mismatch_counts"].values())
    assert shuffled_result["row_order_preserved"] is False

    pd.testing.assert_frame_equal(raw, raw_before)
    pd.testing.assert_frame_equal(normalized, normalized_before)
